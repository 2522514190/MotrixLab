# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate a planted-feet approach to the first largebox pose."""

import argparse
import os
import tempfile
from pathlib import Path

import mujoco as mj
import numpy as np
from opentrack_start import ASSETS, SOURCE, build, fk, joint_addresses, load, pack
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

BASE = ASSETS / "sub3_largebox_003_w_obj.npz"


def planted_prefix(args, a):
    model = mj.MjModel.from_xml_path(str(args.source_xml))
    data = mj.MjData(model)
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    goal = pack(model, a)[0]
    start = pack(model, load(args.input.with_stem(args.input.stem + "_opentrack_ik_start")))[0]
    target_pos, target_quat = fk(model, [goal], feet)
    initial_pos, _ = fk(model, [start], feet)
    start[:3] += (target_pos - initial_pos).mean(axis=(0, 1))
    rotation = Slerp([0, 1], Rotation.from_quat(np.roll([start[3:7], goal[3:7]], -1, axis=1)))
    addresses = joint_addresses(model, a)
    leg = [
        addr for addr, name in zip(addresses, a["joint_names"]) if any(k in name for k in ("hip_", "knee_", "ankle_"))
    ]
    limits = model.jnt_range[[model.joint(a["joint_names"][addresses.index(i)]).id for i in leg]]
    frames = []
    fps = int(a["fps"])
    for t in np.arange(fps) / fps:
        w = t**3 * (10 - 15 * t + 6 * t * t)
        q = (1 - w) * start + w * goal
        q[3:7] = np.roll(rotation(w).as_quat(), 1)

        def residual(x):
            q[leg] = x
            data.qpos[:] = q
            mj.mj_forward(model, data)
            actual = Rotation.from_quat(np.roll(data.xquat[feet], -1, axis=1))
            angle = (actual * Rotation.from_quat(target_quat[0]).inv()).as_rotvec()
            return np.r_[(data.xpos[feet] - target_pos[0]).ravel(), angle.ravel()]

        solution = least_squares(
            residual, frames[-1][leg] if frames else q[leg], bounds=limits.T, gtol=1e-12, ftol=1e-12, xtol=1e-12
        )
        if np.max(np.abs(residual(solution.x))) > 1e-5:
            raise ValueError(f"Unreachable foot target at {t} seconds")
        frames.append(q.copy())
    return np.concatenate((np.repeat(frames[:1], fps, axis=0), frames)), addresses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE)
    args = parser.parse_args()
    args.source_xml = SOURCE
    args.end_hold_seconds = 2
    output = args.input.with_stem(args.input.stem + "_fixed_feet_start")
    out, n = build(args, load(args.input), planted_prefix)
    with tempfile.NamedTemporaryFile(dir=output.parent) as f:
        np.savez_compressed(f, **out)
        f.flush()
        os.link(f.name, output)
    print(output, n, int(out["num_frames"]))
