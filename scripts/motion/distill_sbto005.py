# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Run an isolated SBTO005 depth-student experiment."""

import argparse
import json
import os
from pathlib import Path

import torch

from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import DEPTH, HEIGHT, WIDTH, student_depth
from motrix_envs.locomotion.wbt.g1.sbto005_distill.evaluation import (
    evaluate_camera_dr,
    evaluate_policies,
    log_evaluation_scalars,
)
from motrix_envs.locomotion.wbt.g1.sbto005_distill.policy import STATE_SIZE, Student
from motrix_envs.locomotion.wbt.g1.sbto005_distill.rollout import (
    Rollout,
    Samples,
    configure_motion,
    make_env,
    make_teacher,
    student_terms,
)
from motrix_envs.locomotion.wbt.g1.sbto005_distill.training import (
    TrainingSpec,
    _normalization,
    _update,
    behavior_loss,
    sha256,
    train,
)
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import TASK


def json_report(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def export(checkpoint_path: Path, output: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["student"]
    center = checkpoint["center"]
    half_range = checkpoint["half_range"]
    model = Student(center - half_range, center + half_range).eval()
    model.load_state_dict(checkpoint)
    scripted = torch.jit.script(model)
    scripted.save(str(output))
    loaded = torch.jit.load(str(output))
    generator = torch.Generator().manual_seed(1001)
    for count in (1, 8):
        inputs = (
            torch.randn(count, STATE_SIZE, generator=generator),
            torch.rand(count, 1, HEIGHT, WIDTH, generator=generator),
        )
        with torch.no_grad():
            torch.testing.assert_close(loaded(*inputs), model(*inputs), atol=1e-6, rtol=0)


def check_held_out(teacher_path, student_path, device):
    env = make_env(8, 9032)
    teacher = make_teacher(teacher_path, env, device)
    rollout = Rollout(env, teacher, device, 9032)
    buffer = Samples.allocate(495, 8)
    rollout.collect(buffer, 495, 8, 1.0)
    data = buffer.take(495, 8)
    model = rollout.student
    model.load_state_dict(torch.load(student_path, map_location=device, weights_only=True)["student"])
    values = []
    with torch.no_grad():
        for ids in torch.arange(495 * 8).split(512):
            batch = data.index(ids).to(device)
            delta = (model(batch.state, batch.depth) - batch.action) / model.half_range
            values.append(delta.square().cpu())
    return float(torch.cat(values).mean().sqrt())


def verify(teacher_path: Path, motion_path: Path, output: Path):
    configure_motion(motion_path)
    cpu = torch.device("cpu")
    with torch.no_grad():
        without = make_env(2, 519, camera=False, perturbations=False)
        with_depth = make_env(2, 519, perturbations=False)
        expert = make_teacher(teacher_path, without, cpu)
        rollout = Rollout(with_depth, make_teacher(teacher_path, with_depth, cpu), cpu, 519)
        other_state = without.init_state()
        sample = Samples.allocate(1, 2)
        reset_seen = False
        error = 0.0
        for frame in range(18):
            if frame == 15:
                for env in (with_depth, without):
                    env.state.episode_steps[0] = env._max_episode_steps() - 1
            policy_obs = rollout.state.obs.policy
            depth = student_depth(with_depth.sim_data[DEPTH])
            actions = expert.act(other_state.obs.policy, deterministic=True)
            torch.testing.assert_close(policy_obs, other_state.obs.policy, rtol=0.0, atol=1e-6)
            expected_terms = policy_obs.index_select(1, student_terms(with_depth))
            rollout.collect(sample, 1, 2, 1.0)
            other_state = without.step(actions)
            batch = sample.take(1, 2)
            torch.testing.assert_close(batch.state, expected_terms, rtol=0.0, atol=0.0)
            torch.testing.assert_close(batch.depth, depth.to(torch.float16), rtol=0.0, atol=0.0)
            error = max(error, float((batch.action - actions).abs().max()))
            if frame == 15:
                assert rollout.state.done.tolist() == [True, False]
                assert with_depth.command_terms["motion"].steps[:, 0].tolist() == [0, 16]
                reset_seen = True
        assert reset_seen and error <= 1e-6
        for _ in range(3):
            obs = rollout.state.obs.policy
            expert_label = rollout.teacher.act(obs, deterministic=True)
            proprio = obs.index_select(1, rollout.columns)
            depth = student_depth(with_depth.sim_data[DEPTH])
            executed = rollout.student(proprio, depth)
            rollout.collect(sample, 1, 2, 0.0)
            batch = sample.take(1, 2)
            torch.testing.assert_close(batch.action, expert_label, rtol=0, atol=1e-6)
            torch.testing.assert_close(rollout.state.obs.policy[:, rollout.columns[-29:]], executed, rtol=0, atol=1e-6)
        fixed = Samples.allocate(32, 2)
        rollout.collect(fixed, 32, 2, 1.0)
        fixed = fixed.take(32, 2).to(cpu)
    model = rollout.student.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    _normalization(model, fixed, cpu, False)
    with torch.no_grad():
        before = float(behavior_loss(model(fixed.state, fixed.depth), fixed.action, model.half_range))
    for _ in range(120):
        _update(model, model, optimizer, fixed, cpu)
    with torch.no_grad():
        after = float(behavior_loss(model(fixed.state, fixed.depth), fixed.action, model.half_range))
    assert after <= 0.2 * before
    output.mkdir(parents=True)
    result = {
        "teacher_action_max_abs_difference": error,
        "partial_reset_and_following_sample": "passed",
        "overfit_initial_loss": before,
        "overfit_final_loss": after,
    }
    json_report(output / "verify.json", result)
    print(result, flush=True)


def cloud_spec(overrides: list[str]) -> TrainingSpec:
    # motrix-pipeline passes these as key=value arguments to train_entry.
    opts = dict(item.split("=", maxsplit=1) for item in overrides)
    task = f"{TASK}/motrix.fastsac"
    if opts.pop("task") != task or opts.pop("task.train_backend") != "torch":
        raise ValueError("Cloud distillation expects the SBTO005 Teacher task and the torch backend.")
    for key in ("render", "play"):
        if opts.pop(key, "false") != "false":
            raise ValueError("Cloud distillation runs headless.")
    for key in ("checkpoint.interval", "logging.interval", "logging.backend"):
        opts.pop(key, None)
    num_envs = int(opts.pop("num_envs", "2048"))
    seed = int(opts.pop("seed", "2"))
    if num_envs != 2048 or opts:
        raise ValueError(f"Expected 2048 environments and supported pipeline options, got {num_envs=} {opts=}.")
    teacher = Path(os.environ["MOTRIX_DISTILL_TEACHER"]).resolve()
    motion = Path(os.environ["MOTRIX_SBTO005_MOTION"]).resolve()
    for path, variable in ((teacher, "MOTRIX_DISTILL_TEACHER_SHA256"), (motion, "MOTRIX_DISTILL_MOTION_SHA256")):
        if sha256(path) != os.environ[variable]:
            raise ValueError(f"Checksum mismatch for {path}.")
    return TrainingSpec(
        teacher,
        motion,
        Path("runs/sbto005-distill/cloud").absolute(),
        envs=num_envs,
        learners=2,
        seed=seed,
        bc_width=256,
        dagger_rounds=32,
        handover_rounds=8,
        updates=256,
    )


def local_direct(args, teacher: Path, motion: Path):
    steps, batch_size = 24, 512
    baseline_path = args.baseline.resolve() if args.camera_dr else None
    if baseline_path is not None:
        old = torch.load(baseline_path, map_location="cpu", weights_only=True)
        if old["stage"] != "dagger" or old["step"] != 2000:
            raise ValueError("Camera DR evaluation expects the previously evaluated DAgger round 2000 checkpoint.")
    spec = TrainingSpec(
        teacher,
        motion,
        args.out,
        envs=args.num_envs,
        device=args.device,
        bc_epochs=0,
        dagger_rounds=args.rounds,
        dagger_steps=steps,
        handover_rounds=0,
        updates=steps * args.num_envs // batch_size,
        batch_size=batch_size,
        camera_dr=args.camera_dr,
        baseline=baseline_path,
    )
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    baseline_report = None

    def evaluate(checkpoint: Path, step: int):
        nonlocal baseline_report
        final = step == spec.dagger_rounds
        if final:
            export(checkpoint, spec.output / "student.ts")
        if spec.camera_dr:
            if baseline_report is None:
                baseline_report = evaluate_camera_dr(
                    teacher, baseline_path, args.trials, device, "dagger_2000_baseline"
                )
            report = evaluate_camera_dr(teacher, checkpoint, args.trials, device, "dagger")
            for condition, group in report.items():
                baseline = baseline_report[condition]["dagger_2000_baseline"]
                expected_poses = [row["initial_box_pose"] for row in baseline["trials"]]
                assert [row["initial_box_pose"] for row in group["dagger"]["trials"]] == expected_poses
                group["dagger_2000_baseline"] = baseline
        else:
            report = evaluate_policies(teacher, None, checkpoint, args.trials, device, include_blind=False)
        name = "evaluation.json" if final else f"evaluation-{step}.json"
        json_report(spec.output / name, report)
        log_evaluation_scalars(spec.output, report, step)

    train(spec, evaluate=evaluate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("verify", "local", "local-full", "local-direct", "throughput", "ddp-smoke", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--teacher", type=Path, required=True)
        command.add_argument("--motion", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name in ("local-full", "local-direct", "evaluate"):
            command.add_argument("--trials", type=int, default=200)
        if name == "local-full":
            command.add_argument("--num-envs", type=int, choices=(128, 256, 512, 1024, 2048), required=True)
        if name == "local-direct":
            command.add_argument("--num-envs", type=int, choices=(64, 128, 256, 512, 1024, 2048), default=1024)
            command.add_argument("--rounds", type=int, default=3000)
            command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            command.add_argument("--camera-dr", action="store_true", help="Use visible box resets and depth noise.")
            command.add_argument("--baseline", type=Path, help="Path to the previous DAgger 2000 checkpoint.")
    commands.add_parser("cloud").add_argument("overrides", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "cloud":
        spec = cloud_spec(args.overrides)
        train(spec)
        configure_motion(spec.motion)
        bc, dagger = spec.output / "bc.pt", spec.output / "dagger.pt"
        export(dagger, spec.output / "student.ts")
        report = evaluate_policies(spec.teacher, bc, dagger, 200, torch.device("cuda:0"))
        json_report(spec.output / "evaluation.json", report)
        return

    args.out = args.out.absolute()
    teacher, motion = args.teacher.resolve(), args.motion.resolve()
    configure_motion(motion)
    if args.mode == "local-direct":
        if args.trials < 1 or args.rounds < 1:
            parser.error("--trials and --rounds must be positive.")
        if args.camera_dr and (args.baseline is None or not args.baseline.is_file()):
            parser.error("--camera-dr requires an existing --baseline checkpoint.")
        local_direct(args, teacher, motion)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "verify":
        verify(teacher, motion, args.out)
        return
    if args.mode == "ddp-smoke":
        spec = TrainingSpec(
            teacher,
            motion,
            args.out,
            device="cpu",
            envs=4,
            learners=2,
            bc_steps=8,
            bc_epochs=1,
            dagger_rounds=3,
            dagger_steps=4,
            handover_rounds=2,
            updates=2,
            batch_size=16,
        )
        train(spec)
        return
    if args.mode == "throughput":
        spec = TrainingSpec(
            teacher,
            motion,
            args.out,
            device=device,
            envs=128,
            bc_steps=64,
            bc_epochs=1,
            dagger_rounds=4,
            dagger_steps=64,
            handover_rounds=2,
        )
        train(spec)
        return

    bc, dagger = args.out / "bc.pt", args.out / "dagger.pt"
    trials = args.trials if args.mode in ("local-full", "evaluate") else 20
    if trials < 1:
        parser.error("--trials must be positive.")
    if args.mode in ("local", "local-full"):
        settings = {}
        if args.mode == "local-full":
            bc_steps = max(512, 262144 // args.num_envs)
            dagger_rounds = 4194304 // (64 * args.num_envs)
            settings = dict(
                envs=args.num_envs,
                bc_steps=bc_steps,
                bc_width=262144 // bc_steps,
                dagger_rounds=dagger_rounds,
                handover_rounds=dagger_rounds // 4,
                updates=8192 // dagger_rounds,
            )
        train(TrainingSpec(teacher, motion, args.out, device=device, **settings))
        json_report(
            args.out / "held-out.json",
            {
                "normalized_bc_action_rmse": check_held_out(teacher, bc, torch.device(device)),
            },
        )
        export(dagger, args.out / "student.ts")
    report = evaluate_policies(teacher, bc, dagger, trials, torch.device(device))
    json_report(args.out / "evaluation.json", report)


if __name__ == "__main__":
    main()
