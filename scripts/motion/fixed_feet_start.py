"""Generate a planted-feet approach to the first largebox pose."""
import argparse
import os
import tempfile

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from opentrack_start import ASSETS, SOURCE, build_with_box, fk, joint_addresses, load, pack
import opentrack_start

BASE = ASSETS / "sub3_largebox_003_w_obj.npz"
OLD = BASE.with_stem(BASE.stem + "_opentrack_ik_start")
OUTPUT = BASE.with_stem(BASE.stem + "_fixed_feet_start")


def planted_prefix(args, a):
    model = mj.MjModel.from_xml_path(str(args.source_xml))
    data = mj.MjData(model)
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    goal = pack(model, a)[0]
    start = pack(model, load(OLD))[0]
    target_pos, target_quat = fk(model, [goal], feet)
    initial_pos, _ = fk(model, [start], feet)
    start[:3] += (target_pos - initial_pos).mean(axis=(0, 1))
    rotation = Slerp([0, 1], Rotation.from_quat(np.roll([start[3:7], goal[3:7]], -1, axis=1)))
    addresses = joint_addresses(model, a)
    leg = [addr for addr, name in zip(addresses, a["joint_names"]) if any(k in name for k in ("hip_", "knee_", "ankle_"))]
    limits = model.jnt_range[[model.joint(a["joint_names"][addresses.index(i)]).id for i in leg]]
    frames = []
    fps = int(a["fps"])
    for t in np.arange(fps) / fps:
        w = t**3 * (10 - 15*t + 6*t*t)
        q = (1-w)*start + w*goal
        q[3:7] = np.roll(rotation(w).as_quat(), 1)

        def residual(x):
            q[leg] = x
            data.qpos[:] = q
            mj.mj_forward(model, data)
            actual = Rotation.from_quat(np.roll(data.xquat[feet], -1, axis=1))
            angle = (actual * Rotation.from_quat(target_quat[0]).inv()).as_rotvec()
            return np.r_[(data.xpos[feet]-target_pos[0]).ravel(), angle.ravel()]

        solution = least_squares(residual, frames[-1][leg] if frames else q[leg], bounds=limits.T,
                                 gtol=1e-12, ftol=1e-12, xtol=1e-12)
        if np.max(np.abs(residual(solution.x))) > 1e-5:
            raise ValueError(f"Unreachable foot target at {t} seconds")
        frames.append(q.copy())
    return np.concatenate((np.repeat(frames[:1], fps, axis=0), frames)), addresses


if __name__ == "__main__":
    args = argparse.Namespace(source_xml=SOURCE, end_hold_seconds=2)
    a = load(BASE)
    from unittest.mock import patch
    with patch.object(opentrack_start, "solve", planted_prefix):
        out, n = build_with_box(args, a)
    with tempfile.NamedTemporaryFile(dir=OUTPUT.parent) as f:
        np.savez_compressed(f, **out)
        f.flush()
        os.link(f.name, OUTPUT)
    print(OUTPUT, n, int(out["num_frames"]))
