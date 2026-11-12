#!/usr/bin/env python3
# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0
"""Audit a named robot/box reference without changing it or training dependencies.

Run with the repository venv Python. The force-balance LP is a necessary-condition
test, not a contact-complementarity solver. Its residual variables are diagnostic
slack and are NEVER applied as forces to a simulation.
"""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linprog
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from motrix_env_mujoco.compiler import MuJoCoSceneCompiler
from motrix_envs.locomotion.wbt.g1.omniretarget_fastsac import make_omniretarget_fastsac_cfg


def save_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def intervals(mask, fps):
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [
        {"first_frame": int(g[0]), "last_frame": int(g[-1]), "start_s": g[0] / fps, "end_s": g[-1] / fps}
        for g in groups
        if len(g)
    ]


def support_distance(point, points):
    """Positive outside a ground contact polygon, negative inside (metres)."""
    points = np.unique(np.asarray(points).reshape(-1, 2), axis=0)
    if len(points) == 0:
        return None
    if len(points) >= 3 and np.linalg.matrix_rank(points - points[0], tol=1e-8) == 2:
        hull = ConvexHull(points)
        return float(np.max(hull.equations[:, :2] @ point + hull.equations[:, 2]))
    delta = points - points[0]
    other = points[np.argmax(np.linalg.norm(delta, axis=1))]
    segment = other - points[0]
    factor = np.clip(np.dot(point - points[0], segment) / max(np.dot(segment, segment), 1e-20), 0, 1)
    return float(np.linalg.norm(point - points[0] - factor * segment))


def solve_balance(required, contacts, actuators, force_bounds, tolerance, passive=None, outer=False):
    """Minimize max normalized force-balance error, then peak actuator usage.

    contacts: (nv x condim Jacobian transpose, friction coefficients).
    Pyramidal cone uses signed axis rays. Outer uses a friction box enclosing
    both pyramidal and elliptic cones, so failure is a stronger impossibility test.
    All contact normal ray weights are nonnegative; no root/box actuators exist.
    """
    nv, nu = actuators.shape
    rays = []
    for jac, friction in contacts:
        dim = jac.shape[1]
        local = []
        if dim == 1:
            local = [np.ones(1)]
        elif outer:
            for signs in itertools.product((-1, 1), repeat=dim - 1):
                local.append(np.r_[1.0, np.asarray(signs) * friction[: dim - 1]])
        else:
            for axis in range(1, dim):
                for sign in (-1, 1):
                    ray = np.zeros(dim)
                    ray[0] = 1
                    ray[axis] = sign * friction[axis - 1]
                    local.append(ray)
        rays.extend(jac @ ray for ray in local)
    contact_matrix = np.column_stack(rays) if rays else np.zeros((nv, 0))
    passive_matrix, passive_bounds = passive if passive is not None else (np.zeros((nv, 0)), [])
    matrix = np.column_stack((actuators, contact_matrix, passive_matrix))
    bounds = list(map(tuple, force_bounds)) + [(0, None)] * len(rays) + list(passive_bounds)
    # |required - A*x| <= rho * tolerance.
    inequalities = np.vstack((np.column_stack((matrix, -tolerance)), np.column_stack((-matrix, -tolerance))))
    rhs = np.r_[required, -required]
    objective = np.zeros(matrix.shape[1] + 1)
    objective[-1] = 1
    result = linprog(objective, A_ub=inequalities, b_ub=rhs, bounds=bounds + [(0, None)], method="highs")
    if not result.success:
        raise RuntimeError(f"Force balance solver failed: {result.message}")
    rho = float(result.x[-1])
    # Lexicographic second solve: keep optimal residual and minimize torque use.
    cap = np.max(np.abs(force_bounds), axis=1)
    usage_constraints = np.zeros((2 * nu, matrix.shape[1] + 1))
    usage_constraints[:nu, :nu] = np.eye(nu)
    usage_constraints[nu:, :nu] = -np.eye(nu)
    usage_constraints[:, -1] = -np.r_[cap, cap]
    second = linprog(
        objective,
        A_ub=np.vstack(
            (np.column_stack((matrix, np.zeros(nv))), np.column_stack((-matrix, np.zeros(nv))), usage_constraints)
        ),
        b_ub=np.r_[required + (rho + 1e-6) * tolerance, -required + (rho + 1e-6) * tolerance, np.zeros(2 * nu)],
        bounds=bounds + [(0, 1.000001)],
        method="highs",
    )
    chosen = second.x[:-1] if second.success else result.x[:-1]
    return {
        "rho": rho,
        "residual": required - matrix @ chosen,
        "torque": chosen[:nu],
        "usage": float(np.max(np.abs(chosen[:nu]) / cap)) if nu else 0.0,
        "normal_force_sum": float(chosen[nu : nu + len(rays)].sum()),
        "status": int(result.status),
        "secondary_status": int(second.status),
    }


class Audit:
    def __init__(self, path):
        self.path = path
        with np.load(path, allow_pickle=False) as archive:
            self.clip = {key: archive[key] for key in archive.files}
        a = self.clip
        self.frames = int(a["num_frames"])
        self.fps = float(a["fps"])
        if self.frames < 9 or self.fps <= 0:
            raise ValueError("At least nine frames and a positive fps are required")
        expected = {
            "joint_pos": (self.frames, len(a["joint_names"])),
            "joint_vel": (self.frames, len(a["joint_names"])),
            "body_pos_w": (self.frames, len(a["body_names"]), 3),
            "body_quat_w": (self.frames, len(a["body_names"]), 4),
            "body_lin_vel_w": (self.frames, len(a["body_names"]), 3),
            "body_ang_vel_w": (self.frames, len(a["body_names"]), 3),
            "ext_object_pos_w": (self.frames, 3),
            "ext_object_quat_w": (self.frames, 4),
            "ext_object_lin_vel_w": (self.frames, 3),
            "ext_object_ang_vel_w": (self.frames, 3),
        }
        for key, shape in expected.items():
            if a[key].shape != shape or not np.isfinite(a[key]).all():
                raise ValueError(f"Invalid shape or nonfinite values: {key}")
        for key in ("body_quat_w", "ext_object_quat_w"):
            if np.max(np.abs(np.linalg.norm(a[key], axis=-1) - 1)) > 1e-4:
                raise ValueError(f"Nonunit quaternion: {key}")
        self.cfg = make_omniretarget_fastsac_cfg()
        self.model = MuJoCoSceneCompiler().compile(self.cfg.scene, self.cfg.sim)
        m = self.model
        self.data = mj.MjData(m)
        self.jids = np.array([m.joint(str(name)).id for name in a["joint_names"]])
        self.jq = m.jnt_qposadr[self.jids]
        self.jv = m.jnt_dofadr[self.jids]
        self.bids = np.array([m.body(str(name)).id for name in a["body_names"]])
        self.root = m.body(str(a["root_body_name"])).id
        self.box = m.body("baseLink").id
        self.free = [int(m.body_jntadr[b]) for b in (self.root, self.box)]
        self.fq = [int(m.jnt_qposadr[j]) for j in self.free]
        self.fv = [int(m.jnt_dofadr[j]) for j in self.free]
        self.root_index = list(a["body_names"]).index(str(a["root_body_name"]))
        if m.nu != len(self.jids) or np.any(m.actuator_gear[:, 0] != 1):
            raise ValueError("Audit requires one unit-gear joint actuator per named joint")
        if np.any(m.actuator_trntype != mj.mjtTrn.mjTRN_JOINT):
            raise ValueError("Only direct joint actuators are supported")
        self.av = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.actuators = np.eye(m.nv)[:, self.av]
        self.force_bounds = m.actuator_forcerange.copy()
        if not np.all(m.actuator_forcelimited):
            raise ValueError("Actuator force limits must be specified")
        for i, j in enumerate(m.actuator_trnid[:, 0]):
            if m.jnt_actfrclimited[j]:
                self.force_bounds[i, 0] = max(self.force_bounds[i, 0], m.jnt_actfrcrange[j, 0])
                self.force_bounds[i, 1] = min(self.force_bounds[i, 1], m.jnt_actfrcrange[j, 1])
        self.nominal_margin = m.geom_margin.copy()
        self.nominal_gap = m.geom_gap.copy()
        self.nominal_friction = m.geom_friction.copy()
        self.q = np.tile(m.qpos0, (self.frames, 1))
        self.q[:, self.jq] = a["joint_pos"]
        self.stored_v = np.zeros((self.frames, m.nv))
        self.stored_v[:, self.jv] = a["joint_vel"]
        for b, adr, veladr, pos, quat, lin, ang in (
            (
                self.root,
                self.fq[0],
                self.fv[0],
                a["body_pos_w"][:, self.root_index],
                a["body_quat_w"][:, self.root_index],
                a["body_lin_vel_w"][:, self.root_index],
                a["body_ang_vel_w"][:, self.root_index],
            ),
            (
                self.box,
                self.fq[1],
                self.fv[1],
                a["ext_object_pos_w"],
                a["ext_object_quat_w"],
                a["ext_object_lin_vel_w"],
                a["ext_object_ang_vel_w"],
            ),
        ):
            if m.jnt_type[m.body_jntadr[b]] != mj.mjtJoint.mjJNT_FREE:
                raise ValueError("Robot and box must both have free joints")
            self.q[:, adr : adr + 3] = pos
            self.q[:, adr + 3 : adr + 7] = quat[:, [3, 0, 1, 2]]
            self.stored_v[:, veladr : veladr + 3] = lin
            self.stored_v[:, veladr + 3 : veladr + 6] = Rotation.from_quat(quat).inv().apply(ang)
        self.v = np.empty_like(self.stored_v)
        for t in range(self.frames):
            lo, hi = max(t - 1, 0), min(t + 1, self.frames - 1)
            mj.mj_differentiatePos(m, self.v[t], (hi - lo) / self.fps, self.q[lo], self.q[hi])
        self.acc = np.gradient(self.v, 1 / self.fps, axis=0, edge_order=2)
        self.smooth_v = savgol_filter(self.v, 9, 3, axis=0)
        self.smooth_acc = savgol_filter(self.v, 9, 3, deriv=1, delta=1 / self.fps, axis=0)

    def mass(self, value):
        self.model.body_mass[self.box] = value
        self.model.body_inertia[self.box] = value * np.array([0.02] * 3)
        mj.mj_setConst(self.model, self.data)

    def contact_settings(self, gap, friction_scale=1):
        # Inflate only broad-phase margins; explicit distance filtering below
        # makes gap a candidate tolerance, NOT a force-generating bridge.
        self.model.geom_margin[:] = np.maximum(self.nominal_margin, gap + 1e-6)
        self.model.geom_gap[:] = self.nominal_gap
        self.model.geom_friction[:] = self.nominal_friction
        self.model.geom_friction[:, 0] *= friction_scale

    def prepare(self, t, smooth=False):
        d, m = self.data, self.model
        mj.mj_resetData(m, d)
        d.qpos[:] = self.q[t]
        d.qvel[:] = self.smooth_v[t] if smooth else self.v[t]
        mj.mj_forward(m, d)
        mass = np.empty((m.nv, m.nv))
        mj.mj_fullM(m, mass, d.qM)
        required = mass @ (self.smooth_acc[t] if smooth else self.acc[t]) + d.qfrc_bias - d.qfrc_passive
        return required

    def kind(self, b1, b2):
        names = [self.model.body(b).name for b in (b1, b2)]
        if self.box in (b1, b2):
            other = b2 if b1 == self.box else b1
            return (
                "box_ground"
                if other == 0
                else "hand_box"
                if "wrist" in self.model.body(other).name
                else "robot_box_other"
            )
        if 0 in (b1, b2):
            return "foot_ground" if any("ankle" in name for name in names) else "robot_ground_other"
        return "robot_self"

    def contacts(self, gap):
        m, d = self.model, self.data
        constraints, rows = [], []
        for c in d.contact:
            if c.dist > gap or c.exclude != 0:
                continue
            b1, b2 = [int(m.geom_bodyid[g]) for g in c.geom]
            jp1, jr1, jp2, jr2 = [np.zeros((3, m.nv)) for _ in range(4)]
            mj.mj_jac(m, d, jp1, jr1, c.pos, b1)
            mj.mj_jac(m, d, jp2, jr2, c.pos, b2)
            basis = c.frame.reshape(3, 3)
            relative = basis @ (jp2 - jp1)
            angular = basis @ (jr2 - jr1)
            jac = np.vstack((relative, angular))[: c.dim].T
            constraints.append((jac, c.friction.copy()))
            velocity = relative @ d.qvel
            rows.append(
                {
                    "body1": m.body(b1).name,
                    "body2": m.body(b2).name,
                    "geom1": m.geom(int(c.geom[0])).name,
                    "geom2": m.geom(int(c.geom[1])).name,
                    "kind": self.kind(b1, b2),
                    "distance_m": float(c.dist),
                    "normal_speed_mps": float(velocity[0]),
                    "tangent_speed_mps": float(np.linalg.norm(velocity[1:])),
                    "point_x_m": float(c.pos[0]),
                    "point_y_m": float(c.pos[1]),
                    "point_z_m": float(c.pos[2]),
                    "mu1": float(c.friction[0]),
                    "mu2": float(c.friction[1]),
                    "condim": int(c.dim),
                }
            )
        return constraints, rows

    def balance(self, t, gap=0.002, smooth=False, outer=False, friction_scale=1):
        self.contact_settings(gap, friction_scale)
        required = self.prepare(t, smooth)
        contacts, rows = self.contacts(gap)
        m = self.model
        # Allow bounded dry friction in either direction as a favorable relaxation.
        # Include near-limit joint reactions, also favorably, to avoid false failures.
        columns = [np.eye(m.nv)[:, i] for i in self.jv if m.dof_frictionloss[i] > 0]
        bounds = [(-m.dof_frictionloss[i], m.dof_frictionloss[i]) for i in self.jv if m.dof_frictionloss[i] > 0]
        for j in self.jids:
            if m.jnt_limited[j]:
                for side, sign in ((0, 1), (1, -1)):
                    if abs(self.q[t, m.jnt_qposadr[j]] - m.jnt_range[j, side]) < 1e-5:
                        columns.append(sign * np.eye(m.nv)[:, m.jnt_dofadr[j]])
                        bounds.append((0, None))
        tolerance = np.full(m.nv, 0.25)
        tolerance[self.fv[0] : self.fv[0] + 3] = 5.0
        tolerance[self.fv[0] + 3 : self.fv[0] + 6] = 1.0
        tolerance[self.fv[1] : self.fv[1] + 3] = max(0.02, 0.01 * m.body_mass[self.box] * 9.81)
        tolerance[self.fv[1] + 3 : self.fv[1] + 6] = 0.02
        passive = (np.column_stack(columns), bounds) if columns else None
        result = solve_balance(required, contacts, self.actuators, self.force_bounds, tolerance, passive, outer)
        return result, rows

    def geometry(self):
        a, m, d = self.clip, self.model, self.data
        self.contact_settings(0.002)
        errors, angle_errors, rows, contact_rows = [], [], [], []
        for t in range(self.frames):
            self.prepare(t)
            error = np.linalg.norm(d.xpos[self.bids] - a["body_pos_w"][t], axis=1)
            target_quat = a["body_quat_w"][t][:, [3, 0, 1, 2]].astype(float)
            target_quat /= np.linalg.norm(target_quat, axis=1)[:, None]
            dot = np.abs((d.xquat[self.bids] * target_quat).sum(axis=1))
            errors.append(error)
            angle_errors.append(2 * np.arccos(np.clip(dot, 0, 1)))
            _, contacts = self.contacts(0.002)
            for contact in contacts:
                contact_rows.append({"frame": t, "time_s": t / self.fps, **contact})
            kinds = Counter(c["kind"] for c in contacts)
            joint_viol = np.maximum(
                m.jnt_range[self.jids, 0] - a["joint_pos"][t], a["joint_pos"][t] - m.jnt_range[self.jids, 1]
            )
            rows.append(
                {
                    "frame": t,
                    "time_s": t / self.fps,
                    "fk_max_m": float(error.max()),
                    "limit_violation_rad": float(max(0, joint_viol.max())),
                    "max_joint_acc_rad_s2": float(np.abs(self.acc[t, self.jv]).max()),
                    "smooth_max_joint_acc_rad_s2": float(np.abs(self.smooth_acc[t, self.jv]).max()),
                    "penetration_m": max([max(0, -c["distance_m"]) for c in contacts], default=0),
                    "foot_slip_mps": max(
                        [
                            c["tangent_speed_mps"]
                            for c in contacts
                            if c["kind"] == "foot_ground" and c["distance_m"] <= 0.002
                        ],
                        default=0,
                    ),
                    "foot_ground_contacts": kinds["foot_ground"],
                    "hand_box_contacts": kinds["hand_box"],
                    "other_robot_box_contacts": kinds["robot_box_other"],
                    "box_ground_contacts": kinds["box_ground"],
                    "self_contacts": kinds["robot_self"],
                    "box_z_m": float(a["ext_object_pos_w"][t, 2]),
                }
            )
            for key, kind, body in (
                ("robot_com_outside_foot_support_m", "foot_ground", self.root),
                ("box_com_outside_ground_support_m", "box_ground", self.box),
            ):
                points = [[c["point_x_m"], c["point_y_m"]] for c in contacts if c["kind"] == kind]
                rows[-1][key] = support_distance(d.subtree_com[body, :2], points)
        errors, angle_errors = np.array(errors), np.array(angle_errors)
        body = [
            {
                "body": str(name),
                "max_position_error_m": float(errors[:, i].max()),
                "rms_position_error_m": float(np.sqrt(np.mean(errors[:, i] ** 2))),
                "max_angle_error_rad": float(angle_errors[:, i].max()),
            }
            for i, name in enumerate(a["body_names"])
        ]
        # Relative pivot offset is measured only where a body's joint anchor is zero.
        offsets = []
        names = list(a["body_names"])
        for i, b in enumerate(self.bids):
            if b == 0 or b == self.root:
                continue
            parent = m.body(int(m.body_parentid[b])).name
            if parent not in names:
                continue
            js = range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
            if any(np.linalg.norm(m.jnt_pos[j]) > 1e-6 for j in js):
                continue
            pi = names.index(parent)
            local = (
                Rotation.from_quat(a["body_quat_w"][:, pi]).inv().apply(a["body_pos_w"][:, i] - a["body_pos_w"][:, pi])
            )
            offsets.append(
                {
                    "body": str(a["body_names"][i]),
                    "parent": parent,
                    "model_offset_m": m.body_pos[b].tolist(),
                    "reference_mean_offset_m": local.mean(axis=0).tolist(),
                    "reference_offset_std_m": local.std(axis=0).tolist(),
                }
            )
        speed = np.max(np.abs(self.v), axis=1)
        channel_consistency = {}
        for prefix in ("body", "ext_object"):
            pos = a[f"{prefix}_pos_w"].astype(float)
            quat = a[f"{prefix}_quat_w"].astype(float)
            linear = np.gradient(pos, 1 / self.fps, axis=0)
            angular = np.zeros(pos.shape)
            for t in range(self.frames):
                lo, hi = max(t - 1, 0), min(t + 1, self.frames - 1)
                angular[t] = (
                    Rotation.from_quat(quat[hi].reshape(-1, 4)) * Rotation.from_quat(quat[lo].reshape(-1, 4)).inv()
                ).as_rotvec().reshape(pos[t].shape) / ((hi - lo) / self.fps)
            channel_consistency[prefix] = {
                "linear_velocity_rmse_mps": float(np.sqrt(np.mean((linear - a[f"{prefix}_lin_vel_w"]) ** 2))),
                "angular_velocity_rmse_radps": float(np.sqrt(np.mean((angular - a[f"{prefix}_ang_vel_w"]) ** 2))),
                "angular_velocity_if_local_rmse_radps": float(
                    np.sqrt(
                        np.mean(
                            (
                                Rotation.from_quat(quat.reshape(-1, 4))
                                .inv()
                                .apply(angular.reshape(-1, 3))
                                .reshape(angular.shape)
                                - a[f"{prefix}_ang_vel_w"]
                            )
                            ** 2
                        )
                    )
                ),
                "max_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1))),
            }
        return (
            rows,
            contact_rows,
            {
                "per_body": body,
                "local_pivot_offsets": offsets,
                "channel_consistency": channel_consistency,
                "max_fk_m": float(errors.max()),
                "velocity_rmse_by_dof": np.sqrt(np.mean((self.v - self.stored_v) ** 2, axis=0)).tolist(),
                "joint_velocity_rmse": float(np.sqrt(np.mean((self.v[:, self.jv] - self.stored_v[:, self.jv]) ** 2))),
                "hold_intervals": intervals(speed < 1e-5, self.fps),
                "hard_limit_violating_frames": sum(r["limit_violation_rad"] > 1e-5 for r in rows),
            },
        )

    def rollout(self, start, mass, timestep, torque):
        """0.3 s, interpolated reference PD + feasible-LP feedforward, bounded motors."""
        self.mass(mass)
        m = self.model
        m.geom_margin[:] = self.nominal_margin
        m.geom_gap[:] = self.nominal_gap
        m.geom_friction[:] = self.nominal_friction
        old_dt = m.opt.timestep
        m.opt.timestep = timestep
        d = mj.MjData(m)
        d.qpos[:] = self.q[start]
        d.qvel[:] = self.v[start]
        kp = m.actuator_gainprm[:, 0].copy()
        kd = -m.actuator_biasprm[:, 2].copy()
        # Treat controls as physical torques in this isolated model only.
        gain, bias, biastype = m.actuator_gainprm.copy(), m.actuator_biasprm.copy(), m.actuator_biastype.copy()
        ctrlrange, ctrllimited = m.actuator_ctrlrange.copy(), m.actuator_ctrllimited.copy()
        m.actuator_gainprm[:] = 0
        m.actuator_gainprm[:, 0] = 1
        m.actuator_biasprm[:] = 0
        m.actuator_biastype[:] = mj.mjtBias.mjBIAS_NONE
        m.actuator_ctrllimited[:] = 1
        m.actuator_ctrlrange[:] = self.force_bounds
        history = []
        try:
            for step in range(int(round(0.3 / timestep))):
                phase = min(start + step * timestep * self.fps, self.frames - 1)
                lo, hi = int(phase), min(int(phase) + 1, self.frames - 1)
                alpha = phase - lo
                target = (1 - alpha) * self.q[lo, m.jnt_qposadr[m.actuator_trnid[:, 0]]] + alpha * self.q[
                    hi, m.jnt_qposadr[m.actuator_trnid[:, 0]]
                ]
                velocity = (1 - alpha) * self.v[lo, self.av] + alpha * self.v[hi, self.av]
                ff = (1 - alpha) * torque[lo] + alpha * torque[hi]
                desired = (
                    ff
                    + kp * (target - d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]])
                    + kd * (velocity - d.qvel[self.av])
                )
                d.ctrl[:] = np.clip(desired, self.force_bounds[:, 0], self.force_bounds[:, 1])
                mj.mj_step(m, d)
                phase = min(start + (step + 1) * timestep * self.fps, self.frames - 1)
                lo, hi = int(phase), min(int(phase) + 1, self.frames - 1)
                alpha = phase - lo
                ref = (1 - alpha) * self.q[lo] + alpha * self.q[hi]
                root_error = np.linalg.norm(d.qpos[self.fq[0] : self.fq[0] + 3] - ref[self.fq[0] : self.fq[0] + 3])
                box_error = np.linalg.norm(d.qpos[self.fq[1] : self.fq[1] + 3] - ref[self.fq[1] : self.fq[1] + 3])
                rotation_error = np.zeros(3)
                ref_quat = ref[self.fq[1] + 3 : self.fq[1] + 7].copy()
                ref_quat /= np.linalg.norm(ref_quat)
                mj.mju_subQuat(rotation_error, d.qpos[self.fq[1] + 3 : self.fq[1] + 7], ref_quat)
                history.append(
                    {
                        "mass_kg": mass,
                        "start_frame": start,
                        "dt_s": timestep,
                        "elapsed_s": (step + 1) * timestep,
                        "root_position_error_m": float(root_error),
                        "box_position_error_m": float(box_error),
                        "box_vertical_error_m": float(d.qpos[self.fq[1] + 2] - ref[self.fq[1] + 2]),
                        "box_orientation_error_rad": float(np.linalg.norm(rotation_error)),
                        "saturated_fraction": float(np.mean(np.abs(desired - d.ctrl) > 1e-6)),
                    }
                )
        finally:
            m.opt.timestep = old_dt
            m.actuator_gainprm[:], m.actuator_biasprm[:], m.actuator_biastype[:] = gain, bias, biastype
            m.actuator_ctrlrange[:], m.actuator_ctrllimited[:] = ctrlrange, ctrllimited
        return history


def self_check():
    """Small physical fixtures protect force signs, actuator bounds, and friction."""
    results = {}
    for name, floor, actuator, limit in (
        ("static_supported", True, False, 0),
        ("unsupported_box", False, False, 0),
        ("actuator_over_limit", False, True, 5),
    ):
        body = (
            '<body pos="0 0 1"><joint type="slide" axis="0 0 1" name="lift"/>'
            if actuator
            else '<body pos="0 0 0.099"><freejoint/>'
        )
        xml = "<mujoco><worldbody>" + ('<geom type="plane" size="2 2 .1"/>' if floor else "") + body
        xml += '<geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody>'
        if actuator:
            xml += f'<actuator><motor joint="lift" forcerange="-{limit} {limit}"/></actuator>'
        xml += "</mujoco>"
        m = mj.MjModel.from_xml_string(xml)
        d = mj.MjData(m)
        mj.mj_forward(m, d)
        contacts = []
        for c in d.contact:
            jac = np.zeros((3, m.nv))
            rot = np.zeros_like(jac)
            mj.mj_jac(m, d, jac, rot, c.pos, int(m.geom_bodyid[c.geom[1]]))
            contacts.append(((c.frame.reshape(3, 3) @ jac).T, c.friction))
        actuators = np.eye(m.nv) if actuator else np.zeros((m.nv, 0))
        bounds = np.array([[-limit, limit]]) if actuator else np.empty((0, 2))
        r = solve_balance(d.qfrc_bias.copy(), contacts, actuators, bounds, np.ones(m.nv) * 0.01)
        results[name] = {"rho": r["rho"], "residual": r["residual"].tolist()}
        assert (r["rho"] < 1) == (name == "static_supported"), (name, r)
    # Friction: 20 N horizontal, 10 N normal, mu=.5 cannot balance.
    r = solve_balance(
        np.array([10.0, 20.0, 0.0]),
        [(np.eye(3), np.ones(5) * 0.5)],
        np.zeros((3, 0)),
        np.empty((0, 2)),
        np.ones(3) * 0.01,
        outer=True,
    )
    assert r["rho"] > 1
    results["friction_violation"] = {"rho": r["rho"]}
    return results


def plot_panels(path, panels, fps):
    """Use Matplotlib when available; retain a dependency-free Pillow fallback."""
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    if matplotlib is not None:
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3 * len(panels)), layout="constrained")
        for axis, (title, series) in zip(np.atleast_1d(axes), panels, strict=True):
            for label, values, color in series:
                axis.plot(np.arange(len(values)) / fps, values, label=label, color=color, linewidth=1.1)
            if "rho" in title:
                axis.axhline(1, color="black", linestyle="--", linewidth=0.8, label="failure threshold")
            axis.set_title(title)
            axis.set_xlabel("Time (s), frame = time * fps")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right", fontsize=9)
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        return
    image = Image.new("RGB", (1400, 270 * len(panels)), "white")
    draw = ImageDraw.Draw(image)
    for panel, (title, series) in enumerate(panels):
        left, top, right, bottom = 95, panel * 270 + 40, 1360, panel * 270 + 220
        draw.text((left, top - 26), title, fill="black")
        values = np.concatenate([np.asarray(y) for _, y, _ in series])
        low, high = min(0.0, float(values.min())), max(1e-8, float(values.max()))
        for i in range(5):
            y = bottom - (bottom - top) * i / 4
            draw.line((left, y, right, y), fill="#dddddd")
            draw.text((4, y - 6), f"{low + (high - low) * i / 4:.3g}", fill="black")
        for name, ys, color in series:
            points = [
                (left + i * (right - left) / (len(ys) - 1), bottom - (float(y) - low) / (high - low) * (bottom - top))
                for i, y in enumerate(ys)
            ]
            draw.line(points, fill=color, width=2)
            draw.text((left + 250 * [s[0] for s in series].index(name), bottom + 27), name, fill=color)
        for i in range(9):
            x = left + (right - left) * i / 8
            draw.text((x - 10, bottom + 6), f"{(len(series[0][1]) - 1) / fps * i / 8:.2f}s", fill="black")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--masses", type=float, nargs="+", default=[0.1, 1, 3, 5])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if any(not np.isfinite(m) or m <= 0 for m in args.masses):
        parser.error("Masses must be finite and positive")
    checks = self_check()
    if args.self_check_only:
        print(json.dumps(checks, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.motion.resolve())
    geometry, contacts, kinematics = audit.geometry()
    save_csv(args.output / "geometry.csv", geometry)
    save_csv(args.output / "contacts.csv", contacts)
    m = audit.model
    report = {
        "motion": str(audit.path),
        "sha256": hashlib.sha256(audit.path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mujoco_version": mj.__version__,
        "frames": audit.frames,
        "fps": audit.fps,
        "scene": "g1-29dof-wbt-largebox-omniretarget-fastsac nominal, no domain randomization",
        "model": {
            "nq": m.nq,
            "nv": m.nv,
            "nu": m.nu,
            "cone": int(m.opt.cone),
            "dt_s": float(m.opt.timestep),
            "solver_iterations": int(m.opt.iterations),
            "box_inertia_per_mass": [0.02, 0.02, 0.02],
            "actuator_limits_Nm": audit.force_bounds.tolist(),
            "geom_materials": [
                {
                    "name": m.geom(i).name,
                    "friction": m.geom_friction[i].tolist(),
                    "condim": int(m.geom_condim[i]),
                    "priority": int(m.geom_priority[i]),
                }
                for i in range(m.ngeom)
                if m.geom_contype[i] or m.geom_conaffinity[i]
            ],
        },
        "method": {
            "derivatives": (
                "mj_differentiatePos central; numpy.gradient edge_order=2; "
                "smooth: Savitzky-Golay 9 samples / cubic on velocity"
            ),
            "gap_m": 0.002,
            "rho_tolerance": {
                "robot_force_N_per_axis": 5,
                "robot_moment_Nm_per_axis": 1,
                "joint_Nm": 0.25,
                "box_force_N_per_axis": "max(.02, .01*m*9.81)",
                "box_moment_Nm_per_axis": 0.02,
            },
            "LP": (
                "min infinity normalized balance residual; then min peak torque utilization; "
                "inner pyramidal, outer independent friction bounds"
            ),
            "relaxations": (
                "arbitrary admissible contact forces, no contact compliance/complementarity or slip law; "
                "bounded dry friction with favorable sign; near-limit reactions allowed"
            ),
            "classification": (
                "outer rho>1 is force-balance violation under stated assumptions; "
                "inner rho<=1 is necessary-condition pass only"
            ),
        },
        "self_checks": checks,
        "kinematics": kinematics,
        "masses": [],
        "sensitivity": [],
        "rollouts": [],
    }
    all_rows = []
    torque_rows = []
    rollout_rows = []
    colors = ["#1565c0", "#e65100", "#2e7d32", "#8e24aa"]
    plot_series = []
    # Include key acceleration/penetration peaks, holds, and lift in local tests.
    keyframes = {
        0,
        audit.frames - 16,
        int(np.argmax([r["penetration_m"] for r in geometry])),
        int(np.argmax([r["max_joint_acc_rad_s2"] for r in geometry])),
        int(np.argmax(audit.clip["ext_object_pos_w"][:, 2])),
    }
    for index, mass in enumerate(args.masses):
        audit.mass(mass)
        rows, torques = [], []
        for t in range(audit.frames):
            inner, cs = audit.balance(t)
            outer, _ = audit.balance(t, outer=True)
            residual = inner["residual"]
            rv, bv = audit.fv
            row = {
                "mass_kg": mass,
                "frame": t,
                "time_s": t / audit.fps,
                "inner_rho": inner["rho"],
                "outer_rho": outer["rho"],
                "peak_torque_utilization": inner["usage"],
                "root_force_residual_N": float(np.linalg.norm(residual[rv : rv + 3])),
                "root_moment_residual_Nm": float(np.linalg.norm(residual[rv + 3 : rv + 6])),
                "box_force_residual_N": float(np.linalg.norm(residual[bv : bv + 3])),
                "box_moment_residual_Nm": float(np.linalg.norm(residual[bv + 3 : bv + 6])),
                "joint_residual_max_Nm": float(np.max(np.abs(residual[audit.jv]))),
                "normal_force_sum_N": inner["normal_force_sum"],
                "solver_status": inner["status"],
                "secondary_status": inner["secondary_status"],
                "outer_solver_status": outer["status"],
                "candidate_contacts": len(cs),
            }
            rows.append(row)
            torques.append(inner["torque"])
            torque_rows.append(
                {"mass_kg": mass, "frame": t, **{m.actuator(i).name: float(v) for i, v in enumerate(inner["torque"])}}
            )
            if t % 150 == 0:
                print(
                    f"mass={mass:g} frame={t}/{audit.frames} rho={inner['rho']:.3g} outer={outer['rho']:.3g}",
                    flush=True,
                )
        all_rows.extend(rows)
        rho = np.array([r["outer_rho"] for r in rows])
        inner_rho = np.array([r["inner_rho"] for r in rows])
        worst = int(np.argmax(rho))
        selected = sorted(keyframes | {worst})
        for frame in selected:
            for gap in (0.0, 0.002, 0.005):
                for smooth in (False, True):
                    r, cs = audit.balance(frame, gap=gap, smooth=smooth, outer=True)
                    report["sensitivity"].append(
                        {
                            "mass_kg": mass,
                            "frame": frame,
                            "gap_m": gap,
                            "smooth": smooth,
                            "friction_scale": 1.0,
                            "outer_rho": r["rho"],
                            "contacts": len(cs),
                        }
                    )
            for friction in (0.5, 1.5):
                r, cs = audit.balance(frame, outer=True, friction_scale=friction)
                report["sensitivity"].append(
                    {
                        "mass_kg": mass,
                        "frame": frame,
                        "gap_m": 0.002,
                        "smooth": False,
                        "friction_scale": friction,
                        "outer_rho": r["rho"],
                        "contacts": len(cs),
                    }
                )
        for frame in selected:
            for timestep in (0.005, 0.0025):
                history = audit.rollout(frame, mass, timestep, np.asarray(torques))
                rollout_rows.extend(history)
                report["rollouts"].append(
                    {
                        "mass_kg": mass,
                        "start_frame": frame,
                        "dt_s": timestep,
                        "max_root_error_m": max(h["root_position_error_m"] for h in history),
                        "max_box_error_m": max(h["box_position_error_m"] for h in history),
                        "min_box_vertical_error_m": min(h["box_vertical_error_m"] for h in history),
                        "max_box_orientation_error_rad": max(h["box_orientation_error_rad"] for h in history),
                        "mean_saturated_fraction": float(np.mean([h["saturated_fraction"] for h in history])),
                    }
                )
        usage = np.array([r["peak_torque_utilization"] for r in rows])
        summary = {
            "mass_kg": mass,
            "classification": "存在明确违反约束"
            if np.any(rho > 1)
            else "未发现明确违反但裕量不足"
            if np.any(inner_rho > 1) or np.any(usage > 0.9)
            else "在所测模型与容差内可行（必要条件）",
            "outer_violating_frames": int(np.sum(rho > 1)),
            "inner_violating_frames": int(np.sum(inner_rho > 1)),
            "outer_violation_intervals": intervals(rho > 1, audit.fps),
            "max_outer_rho": float(rho.max()),
            "worst_frame": worst,
            "worst_frame_metrics": rows[worst],
            "peak_torque_utilization": float(usage.max()),
            "near_saturation_frames": int(np.sum(usage > 0.9)),
        }
        report["masses"].append(summary)
        plot_series.append((f"{mass:g} kg outer rho", rho, colors[index % len(colors)]))
        save_csv(args.output / "dynamics.csv", all_rows)
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    save_csv(args.output / "torques.csv", torque_rows)
    save_csv(args.output / "rollouts.csv", rollout_rows)
    save_csv(args.output / "sensitivity.csv", report["sensitivity"])
    plot_panels(
        args.output / "diagnostics.png",
        [
            ("Force balance: outer-cone normalized residual (rho > 1 fails)", plot_series),
            (
                "Joint acceleration (rad/s^2)",
                [
                    ("raw", [r["max_joint_acc_rad_s2"] for r in geometry], colors[0]),
                    ("9-frame smooth", [r["smooth_max_joint_acc_rad_s2"] for r in geometry], colors[1]),
                ],
            ),
            (
                "Geometry (m)",
                [
                    ("penetration", [r["penetration_m"] for r in geometry], colors[1]),
                    ("max FK position mismatch", [r["fk_max_m"] for r in geometry], colors[0]),
                ],
            ),
            (
                "Contact kinematics",
                [
                    ("foot tangential speed m/s", [r["foot_slip_mps"] for r in geometry], colors[0]),
                    ("box height m", [r["box_z_m"] for r in geometry], colors[2]),
                ],
            ),
        ],
        audit.fps,
    )
    write_report(args.output, report, geometry, contacts)
    assert hashlib.sha256(audit.path.read_bytes()).hexdigest() == report["sha256"]
    print(f"Report: {args.output.resolve() / 'REPORT.md'}", flush=True)


def write_report(output, report, geometry, contacts):
    table = "\n".join(
        f"| {s['mass_kg']:g} | {s['classification']} | {s['outer_violating_frames']} / {report['frames']} "
        f"| {s['max_outer_rho']:.2f} | {s['worst_frame']} |"
        for s in report["masses"]
    )
    pairs = {}
    for c in contacts:
        key = (c["geom1"], c["geom2"], c["kind"])
        if key not in pairs or c["distance_m"] < pairs[key]["distance_m"]:
            pairs[key] = c
    penetration = sorted(pairs.values(), key=lambda c: c["distance_m"])[:10]
    ptable = "\n".join(
        f"| {c['geom1']} / {c['geom2']} | {c['kind']} | {c['frame']} | {c['distance_m'] * 1000:.2f} |"
        for c in penetration
    )
    peaks = sorted(geometry, key=lambda r: r["max_joint_acc_rad_s2"], reverse=True)[:8]
    peaktext = ", ".join(f"{r['frame']} ({r['time_s']:.2f}s, {r['max_joint_acc_rad_s2']:.1f} rad/s²)" for r in peaks)
    worst_rollouts = sorted(report["rollouts"], key=lambda r: r["max_box_error_m"], reverse=True)[:8]
    rtable = "\n".join(
        f"| {r['mass_kg']:g} | {r['start_frame']} | {r['dt_s']} | {r['max_root_error_m']:.3f} "
        f"| {r['max_box_error_m']:.3f} | {r['min_box_vertical_error_m']:.3f} |"
        for r in worst_rollouts
    )
    unsupported = [
        r
        for r in geometry
        if not r["hand_box_contacts"] and not r["other_robot_box_contacts"] and not r["box_ground_contacts"]
    ]
    consistency = report["kinematics"]["channel_consistency"]
    angular_table = "\n".join(
        f"| {key} | {v['angular_velocity_rmse_radps']:.6f} | {v['angular_velocity_if_local_rmse_radps']:.6f} |"
        for key, v in consistency.items()
    )
    sensitive = [
        s for s in report["sensitivity"] if s["mass_kg"] == report["masses"][0]["mass_kg"] and s["friction_scale"] == 1
    ]
    stable = "\n".join(
        f"| {frame} | {min(s['outer_rho'] for s in sensitive if s['frame'] == frame):.2f} "
        f"| {max(s['outer_rho'] for s in sensitive if s['frame'] == frame):.2f} |"
        for frame in sorted({s["frame"] for s in sensitive})
    )
    first, last = geometry[0], geometry[-1]
    baseline_rollout = next(
        r
        for r in report["rollouts"]
        if r["mass_kg"] == report["masses"][0]["mass_kg"] and r["start_frame"] == 0 and r["dt_s"] == 0.005
    )
    masses = " ".join(str(s["mass_kg"]) for s in report["masses"])
    reproduction = (
        f".venv/bin/python scripts/motion/diagnose_largebox.py --motion {report['motion']} "
        f"--masses {masses} --output {output.resolve()}"
    )
    text = f"""# 带箱子参考轨迹动力学诊断

## 结论

轨迹在当前名义模型下的精确跟踪需要修复或优化；具体违反区间见 `metrics.json`。优先处理角速度坐标系、腰部模型差异、起止箱子落地姿态、搬运过程的支撑与左手/大腿碰撞。这不等同于策略无法在允许偏差下完成搬运。

| 箱子质量 kg | 判断 | 宽松摩擦锥仍违反帧数 | 最大归一化残差 rho | 最差帧 |
|---|---|---|---|---|
{table}

本报告帧号全部从零开始，时间为 frame / {report["fps"]:g}。rho≤1 只表示满足所测必要条件，不能证明软接触、滑动规律或实机可行；rho>1 表示即使允许有利的接触力和关节干摩擦，也无法在规定残差内平衡。

## 数据、模型与运动学

- 输入：`{report["motion"]}`，{report["frames"]} 帧，{report["fps"]:g} Hz；SHA256：`{report["sha256"]}`。
- MuJoCo {report["mujoco_version"]}，场景 `{report["scene"]}`；机器人与箱子均为浮动基座，29 个直接关节执行器。未执行域随机化、外推力或策略训练。
- 惯量：箱子主惯量为质量 × (0.02, 0.02, 0.02) kg·m²。碰撞网格、质量中心、机器人惯量与关节限位均沿用当前模型。
- 摩擦以编译后的模型和接触合并结果为准，见 `metrics.json` 的 geom_materials 和 `contacts.csv`。不能用 URDF 中的 0.9 注释替代实际值。
- 硬关节限位违反帧数：{report["kinematics"]["hard_limit_violating_frames"]}；关节速度差分一致性 RMSE：{report["kinematics"]["joint_velocity_rmse"]:.6g} rad/s。
- 最大 FK 位置偏差：{report["kinematics"]["max_fk_m"] * 1000:.2f} mm。`local_pivot_offsets` 显示腰部固定偏移差异；参考身体通道和当前模型并非严格一致。因此动力学从根位姿 + 关节位置重建，不强制同时满足矛盾的身体位置通道。
- 保持区间见 `hold_intervals`。主要加速度峰值：{peaktext}。
- 没有任何候选箱子接触的帧数：{len(unsupported)}；这只是几何接触筛查，不代表其余帧均有足够支撑。脚接触点切向速度见 `geometry.csv`，不能仅凭脚踝速度判断脚滑。

### 角速度通道存在坐标系不一致

以四元数差分重建角速度，分别按世界系和身体局部系对照存储值：

| 通道 | 按字段声明世界系 RMSE rad/s | 按局部系 RMSE rad/s |
|---|---|---|
{angular_table}

存储值明显更符合局部系，而字段名/schema 声明为世界系。应追查生成/转换链并按消费接口重算；本次动力学和前向初始化全部从位姿重建速度，因此不依赖这些有歧义的角速度字段。

### 起止保持段的箱子支撑

第 0 帧箱子质心投影距地面候选支撑边约 {first["box_com_outside_ground_support_m"] * 100:.2f} cm，末帧约 {last["box_com_outside_ground_support_m"] * 100:.2f} cm；均无机器人接触，静止保持的精确姿态缺少平衡力矩。这里描述的是初始接触边，不是整个箱底的尺寸。软接触允许箱子稍微转动后形成新接触：基线起点 0.3 s 前向仿真的最大平移只有 {baseline_rollout["max_box_error_m"] * 1000:.2f} mm，最大转角 {baseline_rollout["max_box_orientation_error_rad"]:.3f} rad。因此优先尝试对起止箱子姿态做自然落稳标定，不能将 15 cm 支撑边距离误读成必须平移 15 cm。

`geometry.csv` 同时记录机器人质心与脚底支撑多边形的关系；运动过程中存在惯性和手箱作用力，质心投影越界本身不作为动态不可行判据。

## 接触与穿透

距离为负代表穿透；软接触允许一定穿透，1 cm 阈值用于筛查，不单独作为物理不可行证明。超过 1 cm 的帧数：{sum(r["penetration_m"] > 0.01 for r in geometry)}。

| 接触几何对 | 类型 | 最深帧 | 有符号距离 mm |
|---|---|---|---|
{ptable}

## 动力学方法与边界

求解 M(q)·qacc + bias − passive = B·tau + Jᵀ·f + 干摩擦/限位反力。仅 tau 是关节执行器；机器人基座和箱子没有控制输入。残差只用于诊断，绝不作为虚拟驱动力施加。

主检查候选接触间隙 2 mm，接触法向力非负，内锥使用 MuJoCo 金字塔摩擦锥的轴向生成射线；另求独立切向界的外包锥。执行器使用模型力矩范围与关节总执行器范围的交集。首先最小化归一化最大残差，再在最优残差处最小化最大力矩利用率。

逐轴容差：机器人基座力 5 N、力矩 1 Nm，关节 0.25 Nm，箱子力 max(0.02 N, 1%重量)、箱子力矩 0.02 Nm。`dynamics.csv` 记录各组残差范数（范数可超过逐轴容差）、求解状态、利用率与全部外锥残差；`torques.csv` 保存内锥解。不可行帧中的力矩只属于最佳残差解，不是可执行控制证明。最小最大残差解在各自由度之间折中，分组残差不等于各组独立的最小残差，不能据此单独归因。

导数使用 MuJoCo 流形位置差分和二阶边界梯度；9 帧三阶 Savitzky–Golay 对速度求导仅作敏感性对照，不改轨迹，也不表示平滑后的位姿已可行。选定保持、抬升、峰值和最差帧，对 0/2/5 mm 接触容差、原始/平滑导数及 0.5/1.5 倍滑动摩擦进行检查，结果见 `sensitivity.csv`。

基线质量下，原始/平滑导数与三种间隙组合的外锥残差范围：

| 帧 | 最小 rho | 最大 rho |
|---|---|---|
{stable}

## 短时前向核对

每个质量从选定帧开始仿真 0.3 s，采用力矩受限的参考 PD + 诊断解前馈，插值关节目标；只作用关节力矩，保留真实软接触。以 5 ms 和 2.5 ms 比较，默认求解迭代数沿用任务设置。这里未测试已有 RL 策略，漂移不能单独归因于轨迹。

| 质量 kg | 起始帧 | 步长 s | 最大根位置误差 m | 最大箱子位置误差 m | 最低箱子竖直误差 m |
|---|---|---|---|---|---|
{rtable}

完整结果见 `rollouts.csv` 与 `metrics.json`；负竖直误差意味着低于参考，不单独证明已从手中滑落。

## 优化优先级

1. **必要：统一参考与模型、修复角速度坐标系。** 核对腰部连接偏移、身体坐标系与关节锚点，确定真实机器人使用的模型版本，再重新计算身体状态与速度；不能只修改奖励容差掩盖矛盾。
2. **必要：先处理持续的接触/支撑违反区间。** 起止箱子先做自然落稳标定；搬运段对照动力学区间、手箱接触与箱地接触，调整手与箱子相对位姿、接近/离开时序及脚底支撑。另修复左手与左大腿约 19 mm 穿透。机器人基座与箱子的残差都必须下降。
3. **按定位结果处理局部峰值。** 对主要加速度峰值与起止保持连接处尝试连续速度/加速度过渡或局部减速，再完整复查；平滑不能补偿缺少接触支撑。
4. **裕量提升：负载与摩擦。** 四种质量逐一验收，优先降低近饱和关节需求并扩大稳定支撑。摩擦扫描只改变系数，不代表经过实测标定。

本次没有生成或覆盖优化轨迹。静止支撑、无支撑悬空、执行器超限和摩擦违反四个物理夹具均已运行通过，数值结果见 self_checks。原始轨迹哈希已在运行结束时复核。

复现（在 MotrixLab 根目录执行）：

```bash
{reproduction}
```

![诊断曲线](diagnostics.png)

物理/API 依据：[MuJoCo 动力学与接触](https://mujoco.readthedocs.io/en/stable/computation/index.html)、[MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)。本次复用已安装的 MuJoCo，未安装归档 mjbatch，也未修改训练依赖。
"""  # noqa: E501
    (output / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
