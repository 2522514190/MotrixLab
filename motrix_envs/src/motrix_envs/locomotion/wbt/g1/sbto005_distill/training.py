# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Train one Student from BC and DAgger, with one collector per learner."""

import hashlib
import json
import os
import resource
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import Student
from motrix_envs.locomotion.wbt.g1.sbto005_distill.rollout import (
    Rollout,
    Samples,
    configure_motion,
    make_env,
    make_teacher,
)


@dataclass(frozen=True)
class TrainingSpec:
    teacher: Path
    motion: Path
    output: Path
    envs: int = 32
    learners: int = 1
    device: str = "cuda"
    seed: int = 2
    bc_steps: int = 512
    bc_width: int | None = None
    bc_epochs: int = 4
    dagger_rounds: int = 12
    dagger_steps: int = 64
    handover_rounds: int = 6
    updates: int = 16
    batch_size: int = 512

    def __post_init__(self):
        if self.learners not in (1, 2) or self.envs % self.learners or self.batch_size % (4 * self.learners):
            raise ValueError("Use one or two learners, evenly split environments and quarter minibatches.")
        if not (2 <= self.handover_rounds <= self.dagger_rounds):
            raise ValueError("DAgger handover must span at least two rounds within the run.")
        if self.bc_width is not None and not (0 < self.bc_width <= self.envs // self.learners):
            raise ValueError("BC sample width must fit one collector.")
        if min(self.envs, self.bc_steps, self.bc_epochs, self.dagger_steps, self.updates, self.batch_size) <= 0:
            raise ValueError("Training sizes and budgets must be positive.")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def behavior_loss(prediction, target, half_range):
    return F.smooth_l1_loss(prediction / half_range, target / half_range, beta=0.1)


def _resource_usage(device: torch.device) -> dict:
    return {
        "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cpu_seconds": process_time(),
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
    }


def _device(spec, rank):
    return torch.device(f"cuda:{rank}" if spec.device == "cuda" else "cpu")


def _rollout(spec, rank, device):
    configure_motion(spec.motion)
    seed = spec.seed + rank * 100003
    env = make_env(spec.envs // spec.learners, seed)
    return Rollout(env, make_teacher(spec.teacher, env, device), device, seed + 1)


def _collector(spec, rank, connection, slots, cpu_student):
    device = _device(spec, rank)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rollout = _rollout(spec, rank, device)
    bounds = rollout.env.action_space
    with connection:
        connection.send((bounds.low.tolist(), bounds.high.tolist()))
        while (request := connection.recv()) is not None:
            slot, steps, width, beta = request
            if beta < 1.0:
                rollout.student.load_state_dict(cpu_student.state_dict())
            stats = rollout.collect(slots[slot], steps, width, beta)
            connection.send(stats | _resource_usage(device))


class Collector:
    """Use the same rollout implementation locally and over a shared-memory mailbox."""

    def __init__(self, spec, rank, device):
        self.device = device
        self.width = spec.envs // spec.learners
        self.bc_width = spec.bc_width or self.width
        shared = spec.learners > 1
        self.slots = (
            Samples.allocate(spec.bc_steps, self.bc_width, shared=shared),
            Samples.allocate(spec.dagger_steps, self.width, shared=shared),
        )
        self.process = None
        if not shared:
            self.rollout = _rollout(spec, rank, device)
            self.model = self.rollout.student
            return
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.model = Student(torch.full((29,), -1.0), torch.ones(29)).share_memory()
        self.process = ctx.Process(target=_collector, args=(spec, rank, child, self.slots, self.model))
        self.process.start()
        child.close()
        try:
            low, high = self.connection.recv()
        except BaseException:
            self.close()
            raise
        low, high = torch.tensor(low), torch.tensor(high)
        self.model.center.copy_((low + high) / 2)
        self.model.half_range.copy_((high - low) / 2)

    def collect(self, slot, beta, model=None):
        dest = self.slots[slot]
        steps, width = dest.state.shape[:2]
        if beta < 1.0:
            self.model.load_state_dict(model.state_dict())
        if self.process is None:
            stats = self.rollout.collect(dest, steps, width, beta) | _resource_usage(self.device)
        else:
            self.connection.send((slot, steps, width, beta))
            stats = self.connection.recv()
        return dest.take(steps, width), stats

    def close(self):
        if self.process is not None:
            try:
                self.connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            self.connection.close()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()


def _normalization(model, samples, device, distributed):
    obs = samples.state.to(device, dtype=torch.float64)
    stats = torch.cat([obs.sum(dim=0), obs.square().sum(dim=0), obs.new_tensor([len(obs)])])
    if distributed:
        dist.all_reduce(stats)
    total, squares = stats[:-1].chunk(2)
    mean = total / stats[-1]
    var = (squares / stats[-1] - mean.square()).clamp_min(0.0)
    model.state_mean.copy_(mean)
    model.state_scale.copy_(var.sqrt().clamp_min(1e-3))


def _update(ddp, model, optimizer, batch, device):
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = behavior_loss(ddp(batch.state, batch.depth), batch.action, model.half_range)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def _join(a, b):
    return Samples(torch.cat((a.state, b.state)), torch.cat((a.depth, b.depth)), torch.cat((a.action, b.action)))


def _pin(rank, learners):
    if learners == 2 and hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) >= learners:
            start = len(cpus) * rank // learners
            stop = len(cpus) * (rank + 1) // learners
            os.sched_setaffinity(0, cpus[start:stop])


def _save(output, name, model, optimizer, stage, step):
    temp = output / f".{name}.tmp"
    torch.save({"student": model.state_dict(), "optimizer": optimizer.state_dict(), "stage": stage, "step": step}, temp)
    temp.replace(output / name)


def _rank(rank, spec: TrainingSpec, rendezvous: str | None = None):
    distributed = spec.learners == 2
    _pin(rank, spec.learners)
    device = _device(spec, rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(spec.seed + rank)
    if distributed:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=spec.learners,
        )
    collector = None
    writer = SummaryWriter(spec.output) if rank == 0 else None
    events = (spec.output / f"learner-{rank}.jsonl").open("w", buffering=1)
    rng = torch.Generator().manual_seed(spec.seed + 197 * (rank + 1))

    def log(stage, step, **fields):
        row = {"stage": stage, "step": step, **fields, "learner": _resource_usage(device)}
        events.write(json.dumps(row, allow_nan=False) + "\n")
        if writer is not None:
            message = {key: fields[key] for key in ("beta", "updates", "loss", "env_steps_per_second") if key in fields}
            for key, value in message.items():
                writer.add_scalar(f"{stage}/{key}", value, step)
            print(f"{stage} {step}: {message}", flush=True)

    try:
        collector = Collector(spec, rank, device)
        buffer_model = collector.model
        low, high = buffer_model.center - buffer_model.half_range, buffer_model.center + buffer_model.half_range
        model = Student(low.cpu(), high.cpu()).to(device)
        ddp = (
            DistributedDataParallel(model, device_ids=[rank] if device.type == "cuda" else None)
            if distributed
            else model
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        batch_size = spec.batch_size // spec.learners

        bc, stats = collector.collect(0, 1.0)
        _normalization(model, bc, device, distributed)
        log("bc_collect", 0, collector=stats, retained=len(bc.state))
        for epoch in range(spec.bc_epochs):
            perm = torch.randperm(len(bc.state), generator=rng)
            losses = [_update(ddp, model, optimizer, bc.index(ids), device) for ids in perm.split(batch_size)]
            log("bc", epoch + 1, updates=len(losses), loss=float(torch.stack(losses).mean()))
        if rank == 0:
            _save(spec.output, "bc.pt", model, optimizer, "bc", spec.bc_epochs)

        replay = deque(maxlen=8)
        for round_index in range(spec.dagger_rounds):
            started = perf_counter()
            beta = max(0.0, 1 - round_index / (spec.handover_rounds - 1))
            current, stats = collector.collect(1, beta, model)
            if len(replay) == replay.maxlen:
                replay.popleft()
            replay.append(current.clone())
            losses = []
            for _ in range(spec.updates):
                recent = replay[int(torch.randint(len(replay), (), generator=rng))]
                batch = _join(recent.sample(batch_size * 3 // 4, rng), bc.sample(batch_size // 4, rng))
                losses.append(_update(ddp, model, optimizer, batch, device))
            loss = torch.stack(losses).mean()
            elapsed = torch.tensor(perf_counter() - started, device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= spec.learners
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            log(
                "dagger",
                round_index + 1,
                beta=beta,
                collector=stats,
                updates=spec.updates,
                loss=float(loss),
                env_steps_per_second=stats["steps"] * spec.learners / float(elapsed),
            )
            if rank == 0 and ((round_index + 1) % 8 == 0 or round_index + 1 == spec.dagger_rounds):
                _save(spec.output, "dagger.pt", model, optimizer, "dagger", round_index + 1)

        if distributed:
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            compare = flat.clone()
            dist.broadcast(compare, 0)
            error = (flat - compare).abs().max()
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            torch.testing.assert_close(error, torch.zeros_like(error), rtol=0, atol=1e-6)
            log("replica_parity", spec.dagger_rounds, max_weight_difference=float(error))
    finally:
        if collector is not None:
            collector.close()
        events.close()
        if writer is not None:
            writer.close()
        if distributed:
            dist.destroy_process_group()


def train(spec: TrainingSpec):
    if spec.device == "cuda" and torch.cuda.device_count() < spec.learners:
        raise RuntimeError(f"Expected {spec.learners} CUDA devices.")
    spec.output.mkdir(parents=True)
    metadata = asdict(spec)
    metadata.update(
        teacher_sha256=sha256(spec.teacher),
        motion_sha256=sha256(spec.motion),
        objective="smooth_l1_beta_0.1_normalized_action",
        collectors=spec.learners,
    )
    (spec.output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
    if spec.learners == 1:
        _rank(0, spec)
    else:
        with TemporaryDirectory(prefix="sbto-distill-ddp-") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_rank, args=(spec, rendezvous), nprocs=spec.learners)
