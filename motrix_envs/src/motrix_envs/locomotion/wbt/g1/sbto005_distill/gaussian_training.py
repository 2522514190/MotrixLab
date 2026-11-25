# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""On-policy Gaussian DAgger for the SBTO005 depth Student."""

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import log, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from motrix_rl_builtin.fastsac.networks import build_actor, initialize_actor
from motrix_rl_builtin.fastsac.sbto005_actor import OBS_SIZE
from torch.nn.parallel import DistributedDataParallel

from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import DEPTH, HEIGHT, WIDTH
from motrix_envs.locomotion.wbt.g1.sbto005_distill.depth_noise import RAMP_STEPS, DepthNoise
from motrix_envs.locomotion.wbt.g1.sbto005_distill.evaluation import evaluate_unfiltered, evaluation_scalars
from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import (
    ACTION_SIZE,
    LOG_STD_MAX,
    LOG_STD_MIN,
    STATE_SIZE,
    GaussianStudent,
    Student,
)
from motrix_envs.locomotion.wbt.g1.sbto005_distill.rollout import (
    Samples,
    configure_motion,
    make_env,
    make_student,
    make_teacher,
)
from motrix_envs.locomotion.wbt.g1.sbto005_distill.training import _observation_totals, _pin, _set_normalization, sha256
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import TASK, Sbto005TeacherEnv
from motrix_rl.metrics import EXIT_TIMEOUT, MetricsLogger
from motrix_rl.training_logger import TrainingLogger

ROLLOUT_STEPS = 24
EVALUATION_ROUNDS = (2000, 5000, 10000)


class ScoredEnv(Sbto005TeacherEnv):
    """Track the closed-loop lift and setdown criteria before auto-reset."""

    def init_state(self):
        self.reference = self.command_terms["motion"].clip.extensions["object_pos_w"].data
        self.lift_streak = torch.zeros(self.num_envs, dtype=torch.long)
        self.setdown_streak = torch.zeros_like(self.lift_streak)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool)
        self.set_down = torch.zeros_like(self.lifted)
        self.outcomes = torch.zeros(3, dtype=torch.long)
        return super().init_state()

    def _reset_done_envs(self):
        if not self._initializing_episode_batch:
            state = self._state
            reference = self.reference[(state.episode_steps - 1).clamp(0, len(self.reference) - 1)]
            position = self.sim_data["object_pos"]
            relative_z = position[:, 2] - self.reference[0, 2]
            reference_z = reference[:, 2] - self.reference[0, 2]
            error = (position - reference).norm(dim=-1)
            lifting = (relative_z >= 0.25) & (reference_z >= 0.25) & (error <= 0.20)
            self.lift_streak.add_(1).mul_(lifting)
            self.lifted.logical_or_(self.lift_streak >= 25)
            down = (reference_z <= 0.08) & (relative_z.abs() <= 0.08) & (error <= 0.20)
            slow = self.sim_data["object_lin_vel"].norm(dim=-1) <= 0.20
            self.setdown_streak.add_(1).mul_(self.lifted & down & slow)
            self.set_down.logical_or_(self.setdown_streak >= 15)
            done = state.done
            final_error = (position - self.reference[-1]).norm(dim=-1)
            carried = self.lifted & self.set_down & self._clip_truncated & (final_error < 0.15)
            self.outcomes += torch.stack((done.sum(), (done & self.lifted).sum(), (done & carried).sum()))
            for values in (self.lift_streak, self.setdown_streak, self.lifted, self.set_down):
                values.masked_fill_(done, 0)
        super()._reset_done_envs()


@dataclass(frozen=True)
class GaussianTrainingSpec:
    teacher: Path
    motion: Path
    output: Path
    envs: int = 4096
    learners: int = 2
    rounds: int = 10000
    seed: int = 2
    device: str = "cuda"
    eval_trials: int = 64


class GaussianRollout:
    def __init__(self, spec, rank, device):
        configure_motion(spec.motion)
        seed = spec.seed + rank * 100003
        self.device = device
        self.env = make_env(
            spec.envs // spec.learners, seed, unfiltered_box=True, clean_teacher=True, env_type=ScoredEnv
        )
        self.teacher = make_teacher(spec.teacher, self.env, device)
        self.model = GaussianStudent(make_student(self.env)).to(device).eval()
        self.state = self.env.init_state()
        self.depth_noise = DepthNoise(self.env.num_envs, device, seed + 89471)
        self.action_rng = torch.Generator(device=device).manual_seed(seed + 1)
        self.control_steps = 0

    @torch.no_grad()
    def collect(self, buffer):
        started = perf_counter()
        self.env.outcomes.zero_()
        for step in range(ROLLOUT_STEPS):
            observations = self.state.obs
            clean = observations.policy.to(self.device)
            noisy = observations["student_state"].to(self.device)
            rows = torch.nonzero(self.state.episode_steps == 0).flatten()
            strength = min(1.0, (self.control_steps + step) / RAMP_STEPS)
            depth = self.depth_noise.frame(self.env.sim_data[DEPTH], rows, None, strength)
            expert = self.teacher.act(clean, deterministic=True)
            buffer.state[step].copy_(noisy)
            buffer.depth[step].copy_(depth)
            buffer.action[step].copy_(self.model.label(expert))
            action = self.model.sample(noisy, depth.float(), self.action_rng)
            self.state = self.env.step(action.cpu())
        self.control_steps += ROLLOUT_STEPS
        return self.env.outcomes.tolist(), perf_counter() - started


def _collector(spec, rank, device, pipe, buffer, weights):
    torch.set_num_threads(2)
    with pipe:
        rollout = GaussianRollout(spec, rank, device)
        pipe.send((rollout.model.student.center.cpu(), rollout.model.student.half_range.cpu()))
        while pipe.recv():
            rollout.model.load_state_dict(weights.state_dict())
            pipe.send(rollout.collect(buffer))


def _update(ddp, model, optimizer, samples, device):
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    std_total = torch.zeros_like(total)
    count = len(samples.state)
    # Accumulate the exact full-rollout gradient in bounded chunks.
    for start in range(0, count, 2048):
        end = min(start + 2048, count)
        batch = samples.index(slice(start, end)).to(device)
        synchronize = ddp.no_sync() if ddp is not model and end < count else nullcontext()
        with synchronize:
            mean, log_std = ddp(batch.state, batch.depth)
            residual = (batch.action - mean) * (-log_std).exp()
            loss = (0.5 * residual.square() + log_std + 0.5 * log(2 * pi)).sum(dim=1).sum() / count
            loss.backward()
        total += loss.detach()
        std_total += log_std.detach().exp().sum()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total, std_total / (count * ACTION_SIZE)


def _check_replicas(model):
    parameters = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    replica = parameters.clone()
    dist.broadcast(replica, 0)
    torch.testing.assert_close(replica, parameters, rtol=1e-6, atol=1e-7)


def _checkpoint(path, step, model, optimizer):
    staging = path.with_suffix(".tmp")
    torch.save(
        {
            "stage": "dagger",
            "objective": "tanh_gaussian_nll",
            "step": step,
            "student": model.student.state_dict(),
            "fc_logstd": model.fc_logstd.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        staging,
    )
    staging.replace(path)


@torch.no_grad()
def verify_fastsac_loading(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["student"]
    bias, scale = state["center"], state["half_range"]
    cfg = SimpleNamespace(
        actor_type="sbto005_depth",
        actor_init=str(path),
        use_tanh=True,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
    )
    actor = build_actor(cfg, OBS_SIZE, ACTION_SIZE, scale, bias, "cpu")
    initialize_actor(actor, cfg)
    student = GaussianStudent(Student(bias - scale, bias + scale))
    student.student.load_state_dict(state)
    student.fc_logstd.load_state_dict(checkpoint["fc_logstd"])

    generator = torch.Generator().manual_seed(2026)
    obs = torch.rand((8, OBS_SIZE), generator=generator)
    student_state = obs[:, :STATE_SIZE]
    depth = obs[:, STATE_SIZE:].reshape(-1, 1, HEIGHT, WIDTH)

    action_actor, mean_actor, log_std_actor = actor(obs)
    mean_student, log_std_student = student(student_state, depth)
    torch.testing.assert_close(action_actor, student.student(student_state, depth))
    torch.testing.assert_close(mean_actor, mean_student)
    torch.testing.assert_close(log_std_actor, log_std_student)

    sampled = student.sample(student_state, depth, torch.Generator().manual_seed(1))
    noise = torch.randn(mean_actor.shape, generator=torch.Generator().manual_seed(1))
    expected = (mean_actor + log_std_actor.exp() * noise).tanh() * actor.action_scale + actor.action_bias
    torch.testing.assert_close(sampled, expected, rtol=1e-5, atol=3e-6)


def _train_rank(rank, spec, rendezvous):
    device = torch.device(f"cuda:{rank}" if spec.device == "cuda" else "cpu")
    distributed = spec.learners > 1
    _pin(rank, spec.learners)
    torch.set_num_threads(2)
    torch.manual_seed(spec.seed)
    width = spec.envs // spec.learners
    buffer = Samples.allocate(ROLLOUT_STEPS, width, shared=distributed)
    pipe = worker = None
    metrics = MetricsLogger.from_env(spec.output) if rank == 0 else None
    steps_per_round = spec.envs * ROLLOUT_STEPS
    training_log = TrainingLogger(spec.output, metrics) if metrics is not None else None
    evaluation_log = TrainingLogger(spec.output / "evaluation-tb", metrics) if metrics is not None else None
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
            name = torch.cuda.get_device_name(device)
            print(f"learner {rank}: {name}, free/total memory {torch.cuda.mem_get_info(device)}")
        if metrics is not None:
            metrics.run(
                "running",
                env_name=TASK,
                algo="gaussian_dagger",
                train_backend="torch",
                seed=spec.seed,
                job_type="train",
                primary_axis="env_steps",
                total_steps=spec.rounds * steps_per_round,
                num_envs=spec.envs,
                run_dir=str(spec.output),
            )
        if distributed:
            # Shared CPU weights bridge a learner and its collector. The rollout mailbox stores fp16 depth.
            weights = GaussianStudent(Student(-torch.ones(ACTION_SIZE), torch.ones(ACTION_SIZE))).share_memory()
            ctx = mp.get_context("spawn")
            pipe, remote = ctx.Pipe()
            worker = ctx.Process(target=_collector, args=(spec, rank, device, remote, buffer, weights))
            worker.start()
            remote.close()
            center, half_range = pipe.recv()
            weights.student.center.copy_(center)
            weights.student.half_range.copy_(half_range)
            model = GaussianStudent(Student(center - half_range, center + half_range)).to(device)
            dist.init_process_group(
                "nccl" if device.type == "cuda" else "gloo", init_method=rendezvous, rank=rank, world_size=spec.learners
            )
            ddp = DistributedDataParallel(
                model, device_ids=[rank] if device.type == "cuda" else None, broadcast_buffers=False
            )
        else:
            rollout = GaussianRollout(spec, rank, device)
            model = ddp = rollout.model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        totals = torch.zeros(2 * STATE_SIZE + 1, dtype=torch.float64, device=device)
        counts = torch.zeros(3, dtype=torch.long)
        started = perf_counter()
        if metrics is not None:
            metrics.event("run_started", learners=spec.learners)
        for step in range(1, spec.rounds + 1):
            if distributed:
                weights.load_state_dict(model.state_dict())
                pipe.send(True)
                outcomes, seconds = pipe.recv()
            else:
                outcomes, seconds = rollout.collect(buffer)
            samples = buffer.take(ROLLOUT_STEPS, width)
            stats = _observation_totals(samples, device)
            if distributed:
                dist.all_reduce(stats)
            totals += stats
            _set_normalization(model.student, totals)
            loss, mean_std = _update(ddp, model, optimizer, samples, device)
            counts += torch.tensor(outcomes)
            if step == 1 or step % 10 == 0 or step == spec.rounds:
                gathered = torch.tensor(
                    [float(loss), float(mean_std), seconds, *counts.tolist()], dtype=torch.float64, device=device
                )
                if distributed:
                    dist.all_reduce(gathered)
                loss_sum, mean_std_sum, collect_seconds, episodes, lifts, successes = gathered.tolist()
                if training_log is not None:
                    mean_std = mean_std_sum / spec.learners
                    scalars = {
                        "dagger/nll": loss_sum / spec.learners,
                        "dagger/mean_action_std": mean_std,
                        "dagger/env_steps_per_second": steps_per_round * spec.learners / collect_seconds,
                        "dagger/episodes": episodes,
                    }
                    if episodes:
                        scalars["dagger/carry_success_rate"] = successes / episodes
                        scalars["dagger/stable_lift_rate"] = lifts / episodes
                    training_log.scalars(scalars, step, env_steps=step * steps_per_round)
                    duration = (perf_counter() - started) / step
                    print(
                        f"round {step}/{spec.rounds}: nll={loss_sum / spec.learners:.4f}, "
                        f"mean_std={mean_std:.4f}, episodes={episodes:g}, "
                        f"lifts={lifts:g}, successes={successes:g}, seconds/round={duration:.3f}",
                        flush=True,
                    )
                counts.zero_()
            if distributed and step in (1, spec.rounds):
                _check_replicas(model)
            if step in (1, spec.rounds) or step % 200 == 0:
                checkpoint = spec.output / f"dagger-{step}.pt"
                if training_log is not None:
                    _checkpoint(checkpoint, step, model, optimizer)
                    if step in (1, spec.rounds):
                        verify_fastsac_loading(checkpoint)
                    training_log.event("checkpoint_saved", step, path=str(checkpoint))
                if step in EVALUATION_ROUNDS and spec.eval_trials:
                    if evaluation_log is not None:
                        report = evaluate_unfiltered(checkpoint, spec.eval_trials, device)
                        (spec.output / f"eval-{step}.json").write_text(json.dumps(report, indent=2) + "\n")
                        evaluation_log.scalars(evaluation_scalars(report), step, env_steps=step * steps_per_round)
                    if distributed:
                        dist.barrier()
    except KeyboardInterrupt:
        if metrics is not None:
            metrics.run("killed")
        raise
    except BaseException as exc:
        if metrics is not None:
            metrics.run("failed", reason=f"{type(exc).__name__}: {exc}"[:2048])
        raise
    else:
        if metrics is not None:
            metrics.run("finished")
    finally:
        if training_log is not None:
            training_log.close()
        if evaluation_log is not None:
            evaluation_log.close()
        if metrics is not None:
            metrics.close()
            metrics.join(EXIT_TIMEOUT)
        if pipe is not None:
            try:
                pipe.send(False)
            except (BrokenPipeError, OSError):
                pass
            pipe.close()
            if worker.pid is not None:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
        if dist.is_initialized():
            dist.destroy_process_group()


def train_gaussian(spec):
    if spec.learners not in (1, 2, 4) or spec.envs % spec.learners or min(spec.rounds, spec.envs) <= 0:
        raise ValueError("Use positive training sizes and evenly split the environments across 1, 2, or 4 learners.")
    if spec.device == "cuda" and torch.cuda.device_count() < spec.learners:
        raise RuntimeError(f"Expected at least {spec.learners} GPUs, got {torch.cuda.device_count()}.")
    spec.output.mkdir(parents=True)
    metadata = asdict(spec) | {
        "teacher_sha256": sha256(spec.teacher),
        "motion_sha256": sha256(spec.motion),
        "objective": "tanh_gaussian_nll",
        "rollout_steps": ROLLOUT_STEPS,
        "device_count": torch.cuda.device_count(),
    }
    (spec.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    if spec.learners == 1:
        _train_rank(0, spec, None)
    else:
        with TemporaryDirectory(prefix="sbto005-gaussian-ddp-") as directory:
            mp.spawn(_train_rank, args=(spec, f"file://{directory}/rendezvous"), nprocs=spec.learners)
