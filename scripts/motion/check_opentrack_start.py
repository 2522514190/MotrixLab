# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Check the saved OpenTrack motion on the Motrix G1. Optional UniLab GUI replay."""

import argparse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
from opentrack_start import FIELDS, INPUT, MOTRIX, load, pack
from scipy.spatial.transform import Rotation


def scene(xml, path):
    tree = ET.parse(xml)
    compiler = tree.find("compiler")
    compiler.set("meshdir", str((xml.resolve().parent / compiler.get("meshdir", ".")).resolve()))
    ET.SubElement(tree.find("worldbody"), "geom", name="floor", type="plane", size="0 0 .05", rgba=".45 .48 .50 1")
    tree.write(path)


def check(m, a, base):
    # Find splice even for optional end hold by matching all joint positions.
    candidates = np.flatnonzero(np.all(a["joint_pos"] == base["joint_pos"][0], axis=1))
    n = next(i for i in candidates if np.array_equal(a["joint_pos"][i : i + len(base["joint_pos"])], base["joint_pos"]))
    assert set(base) == set(a) and all(np.isfinite(a[k]).all() for k in FIELDS)
    for k in base:
        assert a[k].dtype == base[k].dtype
        if k != "num_frames":
            np.testing.assert_array_equal(a[k][n : n + len(base[k])] if k in FIELDS else a[k], base[k])
    q = pack(m, a)
    assert len(q) == int(a["num_frames"])
    j = a["joint_pos"]
    jump = np.max(abs(np.diff(j, axis=0)), axis=1)
    rot = Rotation.from_quat(np.roll(q[n - 1 : n + 1, 3:7], -1, axis=1))
    print("prefix / original / total", n, len(base["joint_pos"]), len(q))
    print("joint jumps rad (prefix incl seam / seam / entire)", max(jump[:n]), jump[n - 1], max(jump))
    print("seam pelvis m, rad", np.linalg.norm(q[n, :3] - q[n - 1, :3]), (rot[1] * rot[0].inv()).magnitude())
    print("peak joint speed rad/s prefix, original", abs(a["joint_vel"][:n]).max(), abs(base["joint_vel"]).max())
    d = mj.MjData(m)
    feet = [[m.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)] for side in ("left", "right")]
    gids = np.array(feet)
    depths, probes, leg_contacts, self_contacts, up = [], [], [], [], []
    for row in q:
        d.qpos[:] = row
        mj.mj_forward(m, d)
        axes = d.geom_xmat[gids].reshape(2, 7, 3, 3)[:, :, :, 2]
        ends = (
            d.geom_xpos[gids][:, :, None, :]
            + np.array([-1, 1])[None, None, :, None] * axes[:, :, None, :] * m.geom_size[gids, 1][:, :, None, None]
        )
        depths.append((ends[:, :, :, 2] - m.geom_size[gids, 0][:, :, None]).min(axis=(1, 2)))
        probes.append(ends[:, :, :, :2])
        up.append(d.xmat[m.body("pelvis").id, 8])
        self_min = leg_min = 0.0
        for c in d.contact:
            bodies = m.geom_bodyid[c.geom]
            if all(bodies):
                self_min = min(self_min, c.dist)
                names = [m.body(i).name for i in bodies]
                if all(any(s in b for s in ("hip", "ankle", "knee")) for b in names):
                    leg_min = min(leg_min, c.dist)
        leg_contacts.append(leg_min)
        self_contacts.append(self_min)
    depths, probes = np.array(depths), np.array(probes)
    limit = m.jnt_range[1:]
    violations = np.maximum(q[:, 7:] - limit[:, 1], limit[:, 0] - q[:, 7:]).max(axis=1)
    for name, sl in (("prefix", slice(0, n)), ("original", slice(n, n + len(base["joint_pos"])))):
        print(
            name,
            "sole min L/R m",
            depths[sl].min(axis=0),
            "self / leg min m",
            min(self_contacts[sl]),
            min(leg_contacts[sl]),
            "joint limit excess rad",
            max(0.0, max(violations[sl])),
            "max pelvis tilt deg",
            np.degrees(np.arccos(np.clip(min(up[sl]), -1, 1))),
        )
    # Infer constant hold and the official 0.3 s closing phase from the file.
    moving = np.flatnonzero(np.any(q[:n] != q[0], axis=1))
    hold = moving[0] if len(moving) else n
    close = n - int(0.3 * int(a["fps"]))
    # In the two-step branch upstream inserts three identical leg poses.
    # Upper-body interpolation can continue through that double stance.
    leg_addr = [
        m.joint(name).qposadr[0]
        for name in a["joint_names"]
        if any(part in name for part in ("hip_", "knee_", "ankle_"))
    ]
    legs = q[hold:close, leg_addr]
    repeated = np.flatnonzero(np.all(legs[1:] == legs[:-1], axis=1))
    split = hold + repeated[0] if len(repeated) >= 3 else close - 1
    phases = [(max(hold - 1, 0), split, 0)]
    if split < close - 1:
        phases.append((split + 3, close - 1, 1))
    for first, last, foot in phases:
        print(
            "step support foot (0=left), max horizontal displacement m",
            foot,
            np.linalg.norm(probes[first : last + 1, foot] - probes[first, foot], axis=-1).max(),
        )
    print(
        "closing phase incl seam foot XY displacements L/R m",
        np.linalg.norm(probes[close : n + 1] - probes[close], axis=-1).max(axis=(0, 2, 3)),
    )
    return q


def dynamics(m, q, fps):
    d = mj.MjData(m)
    d.qpos[:] = q[0]
    mj.mj_forward(m, d)
    steps = round(1 / (fps * m.opt.timestep))
    dt = 1 / (fps * steps)
    m.opt.timestep = dt
    adr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    err = []
    fall = None
    for t, row in enumerate(q):
        for s in range(steps):
            alpha = s / steps
            d.ctrl[:] = (1 - alpha) * row[adr] + alpha * q[min(t + 1, len(q) - 1), adr]
            mj.mj_step(m, d)
        err.append(np.linalg.norm(d.qpos[:3] - row[:3]))
        if fall is None and (d.qpos[2] < 0.35 or d.xmat[m.body("pelvis").id, 8] < 0.5):
            fall = t
    print("PD first fall frame (pelvis z<0.35 m or tilt>60 deg), max pelvis error m", fall, max(err))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=INPUT.with_stem(INPUT.stem + "_opentrack_ik_start"))
    ap.add_argument("--original", type=Path, default=INPUT)
    ap.add_argument("--xml", type=Path, default=MOTRIX)
    ap.add_argument("--pd", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--unilab", type=Path, default=Path.home() / "UniLab")
    args = ap.parse_args()
    a = load(args.input)
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "scene.xml"
        scene(args.xml, xml)
        m = mj.MjModel.from_xml_path(str(xml))
        q = check(m, a, load(args.original))
        if args.pd:
            dynamics(m, q, int(a["fps"]))
        if args.view:
            # UniLab expects scalar-first quats and a length-one fps array.
            view = Path(tmp) / "view.npz"
            np.savez(view, **(a | {"fps": a["fps"].reshape(1), "body_quat_w": np.roll(a["body_quat_w"], 1, axis=-1)}))
            assert a["body_names"][1] == "pelvis"
            subprocess.run(
                [
                    args.unilab / ".venv/bin/python",
                    args.unilab / "scripts/motion/replay_npz.py",
                    "--npz_file",
                    view,
                    "--model_file",
                    xml,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
