# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Paired closed-loop evaluation using the SBTO005 carry criteria."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from motrix_env_core.math.quaternion import rotate_inverse
from motrix_envs.locomotion.wbt.g1.common import G1_TRACKED_BODY_NAMES
from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import DEPTH, student_depth
from motrix_envs.locomotion.wbt.g1.sbto005_distill.depth_noise import DepthNoise
from motrix_envs.locomotion.wbt.g1.sbto005_distill.rollout import make_env, make_student, make_teacher, student_terms
from motrix_envs.locomotion.wbt.g1.sbto005_distill.visibility import MIN_BOX_PIXELS


def first_streak(condition: np.ndarray, length: int, after: int = 0) -> int | None:
    hits = np.flatnonzero(np.convolve(condition[after:], np.ones(length, dtype=int), mode="full") == length)
    return int(after + hits[0] - length + 1) if len(hits) else None


def score(position, speed, reference, complete):
    relative_z = position[:, 2] - reference[0, 2]
    reference_z = reference[: len(position), 2] - reference[0, 2]
    error = np.linalg.norm(position - reference[: len(position)], axis=-1)
    lifted = first_streak((relative_z >= 0.25) & (reference_z >= 0.25) & (error <= 0.20), 25)
    down = slip = None
    if lifted is not None:
        after = lifted + 24
        down = first_streak(
            (reference_z <= 0.08) & (np.abs(relative_z) <= 0.08) & (error <= 0.20) & (speed <= 0.20), 15, after
        )
        lost_height = (np.maximum.accumulate(relative_z) - relative_z >= 0.15) & (reference_z >= 0.25)
        slip = first_streak(lost_height & (relative_z < reference_z - 0.15), 5, after)
    final_error = float(np.linalg.norm(position[-1] - reference[-1]))
    success = complete and lifted is not None and down is not None and final_error < 0.15
    if success:
        outcome = "success"
    elif lifted is None:
        outcome = "no_stable_lift"
    elif slip is not None:
        outcome = "slip_during_carry"
    elif not complete:
        outcome = "terminated_after_lift"
    elif down is None:
        outcome = "no_stable_setdown"
    else:
        outcome = "final_position_error"
    return {
        "success": success,
        "outcome": outcome,
        "complete": complete,
        "lift_start": lifted,
        "setdown_start": down,
        "slip_start": slip,
        "final_position_error_m": final_error,
        "peak_lift_m": float(relative_z.max()),
    }


@torch.no_grad()
def evaluate_batch(count, seed, device, teacher_path, student_path, box_xy, box_yaw, blind, *, noise_strength=None):
    visible_box = noise_strength is not None
    env = make_env(count, seed, evaluation=True, box_xy=box_xy, box_yaw=box_yaw, visible_box=visible_box)
    sim = env.sim_data
    state = env.init_state()
    noise = DepthNoise(count, device, seed + 89471) if visible_box else None
    all_rows = torch.arange(count)
    empty_rows = all_rows[:0]
    columns = student_terms(env)
    if student_path is None:
        teacher = make_teacher(teacher_path, env, device)
    else:
        model = make_student(env).to(device).eval()
        model.load_state_dict(torch.load(student_path, map_location=device, weights_only=True)["student"])
    torch.manual_seed(seed + 1)
    initial = torch.cat([sim["object_pos"], sim["object_quat"]], dim=-1).tolist()
    motion = env.command_terms["motion"].clip
    reference = motion.extensions["object_pos_w"].data.numpy()
    base_index = G1_TRACKED_BODY_NAMES.index("pelvis")
    ids = [i for i in range(len(G1_TRACKED_BODY_NAMES)) if i != base_index]
    ref_relative = rotate_inverse(
        motion.tracked_bodies_quat_w[:, base_index, None],
        motion.tracked_bodies_pos_w[:, ids] - motion.tracked_bodies_pos_w[:, base_index, None],
    )
    live = torch.ones(count, dtype=torch.bool)
    lengths = torch.full((count,), len(reference), dtype=torch.long)
    completed = torch.zeros(count, dtype=torch.bool)
    failure_masks = {name: torch.zeros_like(live) for name in env.termination_manager.terms}
    pos, speeds, body_sq = [], [], []
    for frame in range(len(reference)):
        obs = state.obs.policy
        if student_path is None:
            action = teacher.act(obs.to(device), deterministic=True)
        else:
            if noise is not None:
                depth = noise.frame(
                    sim[DEPTH], all_rows if frame == 0 else empty_rows, env.box_pixel_mask, noise_strength
                )
            else:
                depth = student_depth(sim[DEPTH])
            if blind:
                depth.fill_(1)
            action = model(obs.index_select(1, columns).to(device), depth.to(device, dtype=torch.float32))
        state = env.step(action.cpu())
        pos.append(sim["object_pos"].clone())
        speeds.append(sim["object_lin_vel"].norm(dim=-1))
        actual = rotate_inverse(
            sim["distill_base_quat"][:, None],
            sim["tracked_body_pos"][:, ids] - sim["distill_base_pos"][:, None],
        )
        body_sq.append((actual - ref_relative[frame]).square().sum(-1).mean(-1))
        done = live & state.done
        lengths[done] = frame + 1
        completed[done] = state.metrics["episode_clip_end"][done]
        for name, failures in failure_masks.items():
            failures[done] = state.metrics[name][done]
        live &= ~done
        if not bool(live.any()):
            break

    positions, speed, error = (torch.stack(trace).numpy() for trace in (pos, speeds, body_sq))
    if noise is not None:
        clean_pixels = env.box_pixel_mask.flatten(1).sum(dim=1).tolist()
        delivered_pixels = noise.first_box_pixels.tolist()
        first_draw = env.box_first_draw.tolist()
    rows = []
    for lane, length in enumerate(lengths.tolist()):
        row = score(positions[:length, lane], speed[:length, lane], reference, bool(completed[lane]))
        row.update(
            seed=seed,
            lane=lane,
            steps=length,
            initial_box_pose=initial[lane],
            body_relative_rms_m=float(np.sqrt(error[:length, lane].mean())),
            failure_terms=[name for name, mask in failure_masks.items() if mask[lane]],
        )
        if noise is not None:
            row.update(
                first_draw_visible=first_draw[lane],
                first_clean_pixels=clean_pixels[lane],
                first_student_pixels=delivered_pixels[lane],
            )
        rows.append(row)
    return rows


def _summarize(rows: list[dict]) -> dict:
    summary = {
        "successes": sum(row["success"] for row in rows),
        "stable_lifts": sum(row["lift_start"] is not None for row in rows),
        "count": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "median_body_relative_rms_m": float(np.median([row["body_relative_rms_m"] for row in rows])),
        "median_box_final_error_m": float(np.median([row["final_position_error_m"] for row in rows])),
        "trials": rows,
    }
    if "first_student_pixels" in rows[0]:
        summary["first_draw_pass_rate"] = sum(row["first_draw_visible"] for row in rows) / len(rows)
        summary["first_frame_pass_rate"] = sum(row["first_student_pixels"] >= MIN_BOX_PIXELS for row in rows) / len(
            rows
        )
    return summary


def evaluate_policies(
    teacher: Path,
    bc: Path | None,
    dagger: Path,
    trials: int,
    device: torch.device,
    *,
    batch_size: int = 32,
    include_blind: bool = True,
) -> dict:
    report = {}
    policies = [("teacher", None, False)]
    if bc is not None:
        policies.append(("bc", bc, False))
    policies.append(("dagger", dagger, False))
    if include_blind:
        policies.append(("dagger_blind", dagger, True))
    for condition, xy, yaw in (("fixed_box", 0.0, 0.0), ("full_dr", 0.10, 30.0)):
        group = {}
        for name, checkpoint, blind in policies:
            rows = []
            for start in range(0, trials, batch_size):
                count = min(batch_size, trials - start)
                rows.extend(evaluate_batch(count, 1001 + start, device, teacher, checkpoint, xy, yaw, blind))
            if name != "teacher":
                assert [r["initial_box_pose"] for r in rows] == [
                    r["initial_box_pose"] for r in group["teacher"]["trials"]
                ]
            group[name] = _summarize(rows)
            print(condition, name, {key: value for key, value in group[name].items() if key != "trials"}, flush=True)
        report[condition] = group
    return report


def evaluate_camera_dr(teacher: Path, student: Path, trials: int, device: torch.device, policy_name: str) -> dict:
    report = {}
    for condition, strength in (("visible_box_clean", 0.0), ("visible_box_noisy", 1.0)):
        rows = []
        for start in range(0, trials, 32):
            rows.extend(
                evaluate_batch(
                    min(32, trials - start),
                    1001 + start,
                    device,
                    teacher,
                    student,
                    0.0,
                    0.0,
                    False,
                    noise_strength=strength,
                )
            )
        summary = _summarize(rows)
        report[condition] = {policy_name: summary}
        print(condition, policy_name, {k: v for k, v in summary.items() if k != "trials"}, flush=True)
    return report


def log_evaluation_scalars(run_dir: Path, report: dict, step: int) -> None:
    """Keep evaluation events separate from the active learner's event stream."""
    with SummaryWriter(run_dir / "evaluation-tb") as writer:
        for condition, policies in report.items():
            for policy, stats in policies.items():
                prefix = f"eval/{condition}/{policy}"
                count = stats["count"]
                writer.add_scalar(f"{prefix}/carry_success_rate", stats["successes"] / count, step)
                writer.add_scalar(f"{prefix}/stable_lift_rate", stats["stable_lifts"] / count, step)
                writer.add_scalar(f"{prefix}/episodes", count, step)
                for metric in ("median_box_final_error_m", "first_draw_pass_rate", "first_frame_pass_rate"):
                    if metric in stats:
                        writer.add_scalar(f"{prefix}/{metric}", stats[metric], step)
