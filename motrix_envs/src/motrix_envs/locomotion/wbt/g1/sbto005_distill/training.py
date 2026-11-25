# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Train one Student with DAgger, with optional BC initialization."""

import hashlib
import json
import os
import resource
from collections import deque
from collections.abc import Callable
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

from motrix_envs.locomotion.wbt.g1.sbto005_distill import depth_noise, visibility
from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import Student
from motrix_envs.locomotion.wbt.g1.sbto005_distill.rollout import (
    Rollout,
    Samples,
    configure_motion,
    make_env,
    make_teacher,
)

DIRECT_CHECKPOINT_INTERVAL = 200
DIRECT_EVALUATION_ROUNDS = (1000, 3000)
CAMERA_DR_EVALUATION_ROUNDS = (1000, 3000, 5000, 8000, 12000)
REPLAY_ROUNDS = 8


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
    camera_dr: bool = False
    baseline: Path | None = None

    def __post_init__(self):
        if self.learners not in (1, 2) or self.envs % self.learners or self.batch_size % (4 * self.learners):
            raise ValueError("Use one or two learners, evenly split environments and quarter minibatches.")
        if self.bc_epochs == 0:
            if self.learners != 1 or self.handover_rounds != 0:
                raise ValueError("Direct DAgger uses one local learner and no Teacher handover.")
        elif not (2 <= self.handover_rounds <= self.dagger_rounds):
            raise ValueError("DAgger handover must span at least two rounds within the run.")
        if self.camera_dr and self.bc_epochs:
            raise ValueError("Camera DR is supported by direct DAgger.")
        if self.bc_width is not None and not (0 < self.bc_width <= self.envs // self.learners):
            raise ValueError("BC sample width must fit one collector.")
        if (
            self.bc_epochs < 0
            or min(self.envs, self.bc_steps, self.dagger_rounds, self.dagger_steps, self.updates, self.batch_size) <= 0
        ):
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
    env = make_env(spec.envs // spec.learners, seed, visible_box=spec.camera_dr)
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


def _observation_totals(samples, device):
    obs = samples.state.to(device, dtype=torch.float64)
    return torch.cat([obs.sum(dim=0), obs.square().sum(dim=0), obs.new_tensor([len(obs)])])


def _set_normalization(model, stats):
    total, squares = stats[:-1].chunk(2)
    mean = total / stats[-1]
    var = (squares / stats[-1] - mean.square()).clamp_min(0.0)
    model.state_mean.copy_(mean)
    model.state_scale.copy_(var.sqrt().clamp_min(1e-3))


def _normalization(model, samples, device, distributed):
    stats = _observation_totals(samples, device)
    if distributed:
        dist.all_reduce(stats)
    _set_normalization(model, stats)


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
    if learners > 1 and hasattr(os, "sched_getaffinity"):
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

        replay = deque(maxlen=REPLAY_ROUNDS)
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


def _train_direct(spec: TrainingSpec, evaluate: Callable[[Path, int], None] | None):
    device = _device(spec, 0)
    torch.set_num_threads(2)
    rollout = _rollout(spec, 0, device)
    model = rollout.student
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs_totals = torch.zeros(2 * model.state_mean.numel() + 1, dtype=torch.float64, device=device)
    rng = torch.Generator().manual_seed(spec.seed + 197)
    slots = [Samples.allocate(spec.dagger_steps, spec.envs) for _ in range(REPLAY_ROUNDS)]
    replay = [slot.take(spec.dagger_steps, spec.envs) for slot in slots]
    evaluation_rounds = CAMERA_DR_EVALUATION_ROUNDS if spec.camera_dr else DIRECT_EVALUATION_ROUNDS

    with SummaryWriter(spec.output) as writer, (spec.output / "learner-0.jsonl").open("w", buffering=1) as events:
        for index in range(spec.dagger_rounds):
            step = index + 1
            model.eval()
            stats = rollout.collect_student(slots[index % REPLAY_ROUNDS])
            started = perf_counter()
            obs_totals += _observation_totals(replay[index % REPLAY_ROUNDS], device)
            _set_normalization(model, obs_totals)
            model.train()
            losses = []
            for _ in range(spec.updates):
                slot = int(torch.randint(min(step, REPLAY_ROUNDS), (), generator=rng))
                losses.append(_update(model, model, optimizer, replay[slot].sample(spec.batch_size, rng), device))
            loss = float(torch.stack(losses).mean())
            optimize_seconds = perf_counter() - started
            scalars = dict(
                loss=loss,
                beta=0.0,
                updates=spec.updates,
                sample_seconds=stats["seconds"],
                optimize_seconds=optimize_seconds,
                env_steps_per_second=stats["steps"] / (stats["seconds"] + optimize_seconds),
                total_env_steps=step * stats["steps"],
                resets=stats["resets"],
            )
            if spec.camera_dr:
                counts = stats["visibility"]
                scalars.update(
                    depth_noise_strength=stats["depth_noise_strength"],
                    first_draw_pass_rate=counts["first_draw_passes"] / counts["resets"],
                    first_frame_pass_rate=counts["first_frame_passes"] / counts["first_frames"],
                    xy_retries=counts["xy_retries"],
                    pose_retries=counts["pose_retries"],
                    first_frame_mask_fallbacks=counts["first_frame_mask_fallbacks"],
                )
            row = dict(stage="dagger", step=step, **scalars, collector=stats, learner=_resource_usage(device))
            events.write(json.dumps(row, allow_nan=False) + "\n")
            for name, value in scalars.items():
                writer.add_scalar(f"dagger/{name}", value, step)
            if step <= 3 or step % 10 == 0 or step == spec.dagger_rounds:
                print(f"dagger {step}/{spec.dagger_rounds}: {scalars}", flush=True)

            name = "dagger.pt"
            if step % 100 == 0 or step == spec.dagger_rounds:
                _save(spec.output, name, model, optimizer, "dagger", step)
                writer.flush()
            if step % DIRECT_CHECKPOINT_INTERVAL == 0 or step in evaluation_rounds:
                name = f"dagger-{step}.pt"
                _save(spec.output, name, model, optimizer, "dagger", step)
            if evaluate is not None and (step in evaluation_rounds or step == spec.dagger_rounds):
                devices = [device.index] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    evaluate(spec.output / name, step)


def train(spec: TrainingSpec, *, evaluate: Callable[[Path, int], None] | None = None):
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
    if spec.camera_dr:
        metadata.update(
            visibility=dict(
                xy_low_m=visibility.XY_LOW,
                xy_high_m=visibility.XY_HIGH,
                yaw_deg=visibility.YAW_DEG,
                min_box_pixels=visibility.MIN_BOX_PIXELS,
                xy_retries=visibility.XY_RETRIES,
                pose_retries=visibility.POSE_RETRIES,
            ),
            depth_noise=dict(
                gaussian_std_normalized=depth_noise.GAUSSIAN_STD,
                salt_probability=depth_noise.SALT_PROB,
                rectangle_count=depth_noise.RECT_COUNT,
                rectangle_size=depth_noise.RECT_SIZE,
                max_lag=depth_noise.MAX_LAG,
                hold_probability=depth_noise.HOLD_PROB,
                ramp_control_steps=depth_noise.RAMP_STEPS,
            ),
        )
        if spec.baseline is not None:
            metadata["baseline_sha256"] = sha256(spec.baseline)
    (spec.output / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
    if spec.bc_epochs == 0:
        _train_direct(spec, evaluate)
    elif spec.learners == 1:
        _rank(0, spec)
    else:
        with TemporaryDirectory(prefix="sbto-distill-ddp-") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_rank, args=(spec, rendezvous), nprocs=spec.learners)
