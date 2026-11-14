# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate a Motrix v1 motion prefix with unmodified upstream OpenTrack."""

import argparse
import importlib
import inspect
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / ".venv-opentrack/OpenTrack"
REV = "cb9b751993a2483e5d1805a2565ddbfe950c04c9"
ASSETS = ROOT / "motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1"
INPUT = ASSETS / "sub3_largebox_003.npz"
MOTRIX = ROOT / "motrix_robots/src/motrix_robots/assets/g1/g1_29dof.xml"
SOURCE = Path.home() / "holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def joint_addresses(m, a):
    names = a["joint_names"].tolist()
    if m.nq not in (36, 43) or len(set(names)) != 29 or m.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected floating-base 29-joint G1")
    return [int(m.joint(name).qposadr[0]) for name in names]


def pack(m, a):
    root = a["body_names"].tolist().index("pelvis")
    q = np.empty((len(a["joint_pos"]), m.nq))
    q[:, :3] = a["body_pos_w"][:, root]
    q[:, 3:7] = np.roll(a["body_quat_w"][:, root], 1, axis=-1)
    q[:, joint_addresses(m, a)] = a["joint_pos"]
    if m.nq == 43:
        adr = int(m.joint("box_free").qposadr[0])
        q[:, adr:adr+3] = a["ext_object_pos_w"]
        q[:, adr+3:adr+7] = np.roll(a["ext_object_quat_w"], 1, axis=-1)
    return q


def upstream(repo):
    if subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip() != REV:
        raise ValueError(f"Expected OpenTrack {REV}")
    os.environ["GLI_PATH"] = str(repo)
    # Import dataset modules from the checkout without running its RL registry __init__.
    for name in ("track_mj", "track_mj.envs", "track_mj.envs.g1_tracking", "track_mj.utils"):
        mod = ModuleType(name)
        mod.__path__ = [str(repo.joinpath(*name.split(".")))]
        sys.modules[name] = mod
    return importlib.import_module("track_mj.utils.dataset.traj_process")


def solve(args, a):
    import osqp

    p = upstream(args.opentrack.resolve())
    m = mj.MjModel.from_xml_path(str(p.consts.FLAT_TERRAIN_XML))
    q = pack(m, a)
    fps = int(a["fps"])
    info = p.TrajectoryInfo([m.joint(i).name for i in range(m.njnt)], p.TrajectoryModel(m.njnt, m.jnt_type.copy()), fps)
    traj = p.Trajectory(
        info, p.TrajectoryData(qpos=q, qvel=np.zeros((len(q), m.nv)), split_points=np.array([0, len(q)]))
    )
    solver = p.SmoothStartEndTransition(m, traj)
    threshold = 0.1 if args.auto_step else np.inf  # upstream one-step mode
    distance = p.StartEndTransitionHandler(traj, m, solver._default_qpos, 0).compute_step_distance()
    counts = Counter()
    original = osqp.OSQP.solve

    def observe(self, *params, **kwargs):
        result = original(self, *params, **kwargs)
        counts[result.info.status] += 1
        return result

    # Read the final residual at the upstream frame recorder, including plateau exits.
    swing = importlib.import_module("track_mj.utils.dataset.start_end_transition_handler").IK_swing_foot
    lines, first = inspect.getsourcelines(swing)
    record = first + next(i for i, line in enumerate(lines) if "qpos_traj.append(" in line)
    errors = []

    def frame_trace(frame, event, arg):
        if event == "line" and frame.f_lineno == record:
            errors.append(float(frame.f_locals["err_norm"]))
        return frame_trace

    def trace(frame, event, arg):
        return frame_trace if frame.f_code is swing.__code__ else None

    previous = sys.gettrace()
    with patch.object(osqp.OSQP, "solve", observe):
        sys.settrace(trace)
        try:
            result = solver.run_interp(
                default_pose_sec=args.stand_seconds, start_step_sec=args.step_seconds, single_step_th=threshold
            ).data.qpos
        finally:
            sys.settrace(previous)
    print("IK final mixed residuals", np.asarray(errors))
    if set(counts) != {"solved"}:
        raise ValueError(f"QP status {counts}")
    n = len(result) - (len(q) - 1)
    print(f"step distance delta={distance:.8f} m, steps={1 + int(distance > threshold)}, QP={counts}, prefix={n}")

    # Recover upstream's single world transform from corresponding original frame 1.
    def rotations(rows):
        return Rotation.from_quat(np.roll(rows[:, 3:7], -1, axis=1))

    rot = rotations(q[1:2])[0] * rotations(result[n : n + 1])[0].inv()
    shift = q[1, :3] - rot.apply(result[n, :3])
    restored = result.copy()
    restored[:, :3] = rot.apply(result[:, :3]) + shift
    restored[:, 3:7] = np.roll((rot * rotations(result)).as_quat(), 1, axis=1)
    np.testing.assert_allclose(restored[n:, :3], q[1:, :3], atol=1e-6)
    np.testing.assert_allclose(restored[n:, 7:], q[1:, 7:], atol=1e-7)
    assert (rotations(restored[n:]) * rotations(q[1:]).inv()).magnitude().max() < 1e-6
    print("prefix inverse transform xyz, rotvec", shift, rot.as_rotvec())
    return restored[:n], joint_addresses(m, a)


def fk(m, rows, ids):
    d = mj.MjData(m)
    pos, quat = [], []
    for row in rows:
        d.qpos[:] = row
        mj.mj_forward(m, d)
        pos.append(d.xpos[ids].copy())
        quat.append(np.roll(d.xquat[ids], -1, axis=1))
    return np.array(pos), np.array(quat)


OBJECT_FIELDS = {k: k.replace("body_", "ext_object_") for k in FIELDS if k.startswith("body_")}


def build_with_box(args, a):
    box = a["body_names"].tolist().index("largebox_link")
    robot = dict(a)
    for body, obj in OBJECT_FIELDS.items():
        if "vel" not in obj:
            np.testing.assert_allclose(a[body][:, box], a[obj], atol=2e-7)
        robot[body] = np.delete(a[body], box, axis=1)
        del robot[obj]
    robot["body_names"] = np.delete(a["body_names"], box)
    out, n = build(args, robot)
    end = len(out["joint_pos"]) - n - len(a["joint_pos"])
    for body, obj in OBJECT_FIELDS.items():
        before = np.repeat(a[obj][:1], n, axis=0)
        after = np.repeat(a[obj][-1:], end, axis=0)
        if "vel" in obj:
            before.fill(0)
            after.fill(0)
        out[obj] = np.concatenate((before, a[obj], after))
        out[body] = np.concatenate((out[body][:, :box], out[obj][:, None], out[body][:, box:]), axis=1)
        out[body][n : n + len(a[body]), box] = a[body][:, box]
        if end and "vel" not in obj:
            out[body][-end:, box] = a[body][-1, box]
        if "vel" not in obj:
            out[body][:n, box] = a[body][0, box]
    out["body_names"] = a["body_names"]
    return out, n


def build(args, a):
    if "largebox_link" in a["body_names"]:
        return build_with_box(args, a)
    if {k for k in a if a[k].ndim and len(a[k]) == len(a["joint_pos"])} != set(FIELDS):
        raise ValueError("Unsupported motion channels; expected robot-only v1")
    fps = int(a["fps"])
    m = mj.MjModel.from_xml_path(str(args.source_xml))
    ids = [m.body(k).id for k in a["body_names"]]
    original = pack(m, a)
    pos, quat = fk(m, original, ids)
    np.testing.assert_allclose(pos, a["body_pos_w"], atol=2e-6)
    np.testing.assert_allclose(abs((quat * a["body_quat_w"]).sum(-1)), 1, atol=2e-6)
    prefix, adr = solve(args, a)
    q = np.repeat(original[:1], len(prefix), axis=0)
    q[:, :7] = prefix[:, :7]
    q[:, joint_addresses(m, a)] = prefix[:, adr]
    pos, quat = fk(m, np.vstack([q, original[:1]]), ids)
    # Choose quaternion signs continuously back from the untouched splice.
    quat[-1] = a["body_quat_w"][0]
    for t in reversed(range(len(q))):
        quat[t] *= np.where((quat[t] * quat[t + 1]).sum(-1) < 0, -1, 1)[:, None]
    rots = Rotation.from_quat(quat[1:].reshape(-1, 4)) * Rotation.from_quat(quat[:-1].reshape(-1, 4)).inv()
    joints = np.vstack([prefix[:, adr], a["joint_pos"][:1]])
    fields = (
        joints[:-1],
        np.diff(joints, axis=0) * fps,
        pos[:-1],
        quat[:-1],
        np.diff(pos, axis=0) * fps,
        rots.as_rotvec().reshape(pos[:-1].shape) * fps,
    )
    out = dict(a)
    end = round(args.end_hold_seconds * fps)
    for k, values in zip(FIELDS, fields):
        hold = np.repeat(a[k][-1:], end, axis=0)
        if "vel" in k:
            hold.fill(0)
        out[k] = np.concatenate([values.astype(a[k].dtype), a[k], hold])
        if not np.isfinite(out[k]).all():
            raise ValueError(k)
    out["num_frames"] = np.asarray(len(out["joint_pos"]), dtype=a["num_frames"].dtype)
    return out, len(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=INPUT)
    ap.add_argument("--output", type=Path, default=INPUT.with_stem(INPUT.stem + "_opentrack_ik_start"))
    ap.add_argument("--source-xml", type=Path, default=SOURCE, help="FK model matching every input body")
    ap.add_argument("--opentrack", type=Path, default=REPO)
    ap.add_argument("--stand-seconds", type=float, default=1.0)
    ap.add_argument("--step-seconds", type=float, default=0.3)
    ap.add_argument("--end-hold-seconds", type=float, default=0.0)
    ap.add_argument(
        "--auto-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Official step-count selection; disabling selects one step",
    )
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    a = load(args.input)
    if min(args.stand_seconds, args.end_hold_seconds) < 0 or args.step_seconds * int(a["fps"]) < 1:
        ap.error("Invalid duration")
    out, n = build(args, a)
    with tempfile.NamedTemporaryFile(dir=args.output.parent) as f:
        np.savez_compressed(f, **out)
        f.flush()
        saved = load(f.name)
        for k in FIELDS:
            np.testing.assert_array_equal(saved[k][n : n + len(a[k])], a[k])
        os.link(f.name, args.output)
    print(args.output, int(out["num_frames"]))


if __name__ == "__main__":
    main()
