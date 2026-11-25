# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Export and evaluate the FastSAC SBTO 005 teacher in native MuJoCo."""

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from math import isfinite
from pathlib import Path

import mujoco as mj
import numpy as np

BUNDLE_VERSION = 1


def export_bundle(run_dir: Path, motion: Path, bundle: Path) -> None:
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler
    from motrix_envs.locomotion.wbt.g1.sbto005_teacher import (
        BOX_GEOM,
        BOX_LINK,
        HAND_GEOMS,
        MOTION_ENV_VAR,
        TASK,
        Sbto005TeacherEnv,
        make_sbto005_teacher_cfg,
    )
    from motrix_rl.checkpoints import best_policy
    from motrix_rl.deploy.onnx import export_onnx
    from motrix_rl.runs import open_run_context

    if bundle.exists():
        raise FileExistsError(f"Choose a new deployment bundle directory: {bundle}")
    run = open_run_context(run_dir)
    metadata = run.metadata
    identity = (metadata.env_name, metadata.rllib, metadata.train_backend, metadata.algo)
    if identity != (TASK, "motrix", "torch", "fastsac"):
        raise ValueError(f"Expected a Motrix Torch FastSAC run for {TASK}: {run_dir}")

    os.environ[MOTION_ENV_VAR] = str(motion.resolve())
    cfg = make_sbto005_teacher_cfg()
    env = Sbto005TeacherEnv(cfg, backend=metadata.sim, headless=True, seed=0)
    term = env.action_terms["joint_position"]
    state = term.state
    joint_names = tuple(actuator.target_name for actuator in env.action_actuators["joint_position"])
    joint_config = {
        "default": state.default_angles.tolist(),
        "scale": state.action_scales.tolist(),
        "limits": [state.joint_lower.tolist(), state.joint_upper.tolist()],
        "action_limits": [term.action_space.low.tolist(), term.action_space.high.tolist()],
        "target_limits": [bound.tolist() for bound in term._target_limits],
    }
    layout = env.observation_manager.groups["policy"]
    observation_terms = tuple(t.name for t in layout.terms)
    expected_terms = (
        "motion_joint",
        "motion_ref_ori_b",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "actions",
        "box_pose",
        "box_pose_error",
    )
    if observation_terms != expected_terms or layout.history_length != 1 or state.delay_hi:
        raise ValueError("The trained policy observation layout or action delay is unsupported.")

    spec = MuJoCoSceneCompiler().create_spec(cfg.scene, cfg.sim)
    spec.body(BOX_LINK).add_freejoint()
    model = spec.compile()
    control_steps = round(cfg.ctrl_dt / model.opt.timestep)
    if not np.isclose(control_steps * model.opt.timestep, cfg.ctrl_dt):
        raise ValueError("MuJoCo timestep must divide the policy timestep.")

    with np.load(motion, allow_pickle=False) as ref:
        if not np.isclose(1 / int(ref["fps"]), cfg.ctrl_dt):
            raise ValueError("The reference frame rate must equal the policy control rate.")
        if tuple(ref["joint_names"]) != joint_names:
            raise ValueError("Reference joints must follow the teacher's actuator order.")

    reset = cfg.sim_reset
    policy = cfg.observations.policy
    terminations = cfg.terminations
    config = {
        "version": BUNDLE_VERSION,
        "task": TASK,
        "source_checkpoint": str(best_policy(metadata, run.run_dir).resolve()),
        "mujoco_version": mj.__version__,
        "control_steps": control_steps,
        "joints": joint_names,
        "joint_config": joint_config,
        "box_body": BOX_LINK,
        "box_geom": BOX_GEOM,
        "hand_geoms": HAND_GEOMS,
        "end_effectors": terminations.bad_body_z.body_names,
        "termination_thresholds": {
            "bad_ref_z": terminations.bad_ref_z.threshold,
            "bad_ref_ori": terminations.bad_ref_ori.threshold,
            "bad_body_z": terminations.bad_body_z.threshold,
        },
        "domain": asdict(cfg.randomization),
        "box_xy_dr_m": reset.object_state.initial_xy_noise,
        "reset_noise": {
            key: (np.asarray(getattr(reset, key).noise) * getattr(reset, key).noise_scale).tolist()
            for key in ("body_pos", "body_rot", "body_lin_vel", "body_rot_vel", "body_dof_pos")
        },
        "observation_noise": {
            key: getattr(policy, key).noise.amplitude
            for key in ("motion_ref_ori_b", "base_ang_vel", "dof_pos", "dof_vel")
        }
        | {"box_pose": policy.box_pose.noise, "box_pose_error": policy.box_pose_error.noise},
        "base_angular_velocity_scale": policy.base_ang_vel.scale,
    }
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bundle.parent, prefix=f".{bundle.name}-") as directory:
        staging = Path(directory) / "bundle"
        artifact = export_onnx(run_dir, staging / "policy.onnx")
        tensor_sizes = (artifact.report.input_spec.shape[-1], artifact.report.output_spec.shape[-1])
        if tensor_sizes != (layout.size, len(joint_names)):
            raise ValueError("The exported actor dimensions do not match the teacher task.")
        mj.mj_saveModel(model, str(staging / "scene.mjb"))
        shutil.copyfile(motion, staging / "motion.npz")
        config["onnx_parity"] = asdict(artifact.report.parity)
        (staging / "deployment.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        staging.rename(bundle)
    print(f"Exported {bundle.resolve()} from {config['source_checkpoint']}")
    print(f"ONNX maximum absolute action error {artifact.report.parity.max_abs_error:.3g}")


def matrix(quat: np.ndarray) -> np.ndarray:
    rotation = np.empty(9)
    mj.mju_quat2Mat(rotation, np.asarray(quat, dtype=np.float64))
    return rotation.reshape(3, 3)


def free_joint_addresses(model: mj.MjModel, body_id: int) -> tuple[int, int]:
    joint = model.body_jntadr[body_id]
    if joint < 0 or model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{model.body(body_id).name} must have a free joint.")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


class TeacherPlayback:
    """Actuate an exported teacher and measure each uninterrupted frame-zero trial."""

    def __init__(self, bundle: Path, config: dict, motion: dict, dr: bool, seed: int):
        self.config = config
        self.motion = motion
        self.dr = dr
        self.rng = np.random.default_rng(seed)
        self.box_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.model = model = mj.MjModel.from_binary_path(str(bundle / "scene.mjb"))
        self.data = mj.MjData(model)
        self.dt = config["control_steps"] * model.opt.timestep
        self.joint_config = {
            name: np.asarray(value, dtype=np.float32) for name, value in config["joint_config"].items()
        }
        joint_ids = np.array([model.joint(name).id for name in config["joints"]])
        actuators_by_joint = {joint: i for i, joint in enumerate(model.actuator_trnid[:, 0])}
        self.actuator_ids = np.array([actuators_by_joint[joint] for joint in joint_ids])
        self.joint_qpos = model.jnt_qposadr[joint_ids]
        self.joint_qvel = model.jnt_dofadr[joint_ids]

        self.base = model.body(str(motion["root_body_name"])).id
        self.torso = model.body(str(motion["reference_body_name"])).id
        self.box = model.body(config["box_body"]).id
        self.base_qpos, self.base_qvel = free_joint_addresses(model, self.base)
        self.box_qpos, self.box_qvel = free_joint_addresses(model, self.box)
        self.box_geom = model.geom(config["box_geom"]).id
        self.hand_geoms = tuple(model.geom(geom).id for geom in config["hand_geoms"])
        body_names = list(motion["body_names"])
        self.ref_base = body_names.index(str(motion["root_body_name"]))
        self.ref_torso = body_names.index(str(motion["reference_body_name"]))
        self.end_effectors = np.array([model.body(name).id for name in config["end_effectors"]])
        self.ref_end_effectors = np.array([body_names.index(name) for name in config["end_effectors"]])
        self.ref_torso_matrix = np.stack([matrix(q) for q in motion["body_quat_w"][:, self.ref_torso]])
        self.ref_box_matrix = np.stack([matrix(q) for q in motion["ext_object_quat_w"]])
        self.previous_action = np.zeros(len(joint_ids), dtype=np.float32)
        self.domain_sample = self.apply_domain()
        self.reset()

    def apply_domain(self) -> dict:
        cfg = self.config["domain"]
        if not (self.dr and cfg["enabled"]):
            return {}
        floor_geom = self.model.geom("floor").id
        floor_friction = self.rng.uniform(*cfg["ground_sliding_friction"])
        box_friction = self.rng.uniform(*cfg["box_sliding_friction"])
        mass_scale = self.rng.uniform(*cfg["box_mass_scale"])
        offset = self.rng.uniform(-cfg["torso_com_offset"], cfg["torso_com_offset"], size=3)
        self.model.geom_friction[floor_geom, 0] = floor_friction
        self.model.geom_friction[self.box_geom, 0] = box_friction
        self.model.body_mass[self.box] *= mass_scale
        self.model.body_inertia[self.box] *= mass_scale
        self.model.body_ipos[self.torso] += offset
        mj.mj_setConst(self.model, self.data)
        return {
            "floor_friction": floor_friction,
            "box_friction": box_friction,
            "box_mass_inertia_scale": mass_scale,
            "torso_com_offset_m": offset.tolist(),
        }

    def reset(self) -> None:
        mj.mj_resetData(self.model, self.data)
        self.previous_action.fill(0)
        motion, data = self.motion, self.data
        noise = self.config["reset_noise"]

        def sample(name: str, size: int = 3) -> np.ndarray:
            return self.rng.uniform(-1, 1, size=size) * noise[name] if self.dr else np.zeros(size)

        qpos, qvel = self.base_qpos, self.base_qvel
        ref = (0, self.ref_base)
        quat_noise = np.empty(4)
        quat = np.empty(4)
        # Extrinsic roll/pitch/yaw, left-multiplied onto the reference.
        mj.mju_euler2Quat(quat_noise, sample("body_rot"), "XYZ")
        mj.mju_mulQuat(quat, quat_noise, motion["body_quat_w"][ref].astype(np.float64))
        data.qpos[qpos : qpos + 3] = motion["body_pos_w"][ref] + sample("body_pos")
        data.qpos[qpos + 3 : qpos + 7] = quat / np.linalg.norm(quat)
        data.qvel[qvel : qvel + 3] = motion["body_lin_vel_w"][ref] + sample("body_lin_vel")
        data.qvel[qvel + 3 : qvel + 6] = matrix(quat).T @ (motion["body_ang_vel_w"][ref] + sample("body_rot_vel"))
        data.qpos[self.joint_qpos] = np.clip(
            motion["joint_pos"][0] + sample("body_dof_pos", self.joint_qpos.size), *self.joint_config["limits"]
        )
        data.qvel[self.joint_qvel] = motion["joint_vel"][0]
        data.qpos[self.box_qpos : self.box_qpos + 3] = motion["ext_object_pos_w"][0]
        offset = self.config.get("box_xy_offset_m")
        if offset is None and self.dr and (amplitude := self.config.get("box_xy_dr_m", 0.0)):
            offset = self.box_rng.uniform(-amplitude, amplitude, size=2)
        if offset is not None:
            data.qpos[self.box_qpos : self.box_qpos + 2] += offset
            self.domain_sample["box_initial_offset_xy_m"] = list(offset)
        data.qpos[self.box_qpos + 3 : self.box_qpos + 7] = motion["ext_object_quat_w"][0]
        data.qvel[self.box_qvel : self.box_qvel + 3] = motion["ext_object_lin_vel_w"][0]
        data.ctrl[self.actuator_ids] = self.target(self.previous_action)
        mj.mj_forward(self.model, data)

    def observation(self, frame: int) -> np.ndarray:
        data, motion = self.data, self.motion
        torso_inverse = data.xmat[self.torso].reshape(3, 3).T
        box_position = data.xpos[self.box]
        box_local = torso_inverse @ data.xmat[self.box].reshape(3, 3)
        desired_box_local = torso_inverse @ self.ref_box_matrix[frame]
        noisy_terms = {
            "motion_ref_ori_b": (torso_inverse @ self.ref_torso_matrix[frame])[:2].ravel(),
            "base_ang_vel": data.qvel[self.base_qvel + 3 : self.base_qvel + 6]
            * self.config["base_angular_velocity_scale"],
            "dof_pos": data.qpos[self.joint_qpos] - self.joint_config["default"],
            "dof_vel": data.qvel[self.joint_qvel],
            "box_pose": np.r_[torso_inverse @ (box_position - data.xpos[self.torso]), box_local[:2].ravel()],
            "box_pose_error": np.r_[
                torso_inverse @ (motion["ext_object_pos_w"][frame] - box_position),
                (desired_box_local @ box_local.T)[:2].ravel(),
            ],
        }
        if self.dr:
            for key, value in noisy_terms.items():
                amplitude = self.config["observation_noise"][key]
                value += self.rng.uniform(-amplitude, amplitude, size=value.shape)
        return np.r_[
            motion["joint_pos"][frame],
            motion["joint_vel"][frame],
            noisy_terms["motion_ref_ori_b"],
            noisy_terms["base_ang_vel"],
            noisy_terms["dof_pos"],
            noisy_terms["dof_vel"],
            self.previous_action,
            noisy_terms["box_pose"],
            noisy_terms["box_pose_error"],
        ].astype(np.float32)

    def target(self, action: np.ndarray) -> np.ndarray:
        cfg = self.joint_config
        return np.clip(cfg["default"] + action * cfg["scale"], *cfg["target_limits"])

    def step(self, action: np.ndarray) -> None:
        self.previous_action[:] = np.clip(action, *self.joint_config["action_limits"])
        self.data.ctrl[self.actuator_ids] = self.target(self.previous_action)
        mj.mj_step(self.model, self.data, nstep=self.config["control_steps"])
        mj.mj_forward(self.model, self.data)

    def failures(self, frame: int) -> list[str]:
        ref_pos = self.motion["body_pos_w"][frame]
        ref_torso_z = ref_pos[self.ref_torso, 2]
        # Training aligns end effectors by yaw and XY while preserving the reference height.
        errors = {
            "bad_ref_z": abs(self.data.xpos[self.torso, 2] - ref_torso_z),
            "bad_ref_ori": abs(self.data.xmat[self.torso, 8] - self.ref_torso_matrix[frame, 2, 2]),
            "bad_body_z": np.max(np.abs(ref_pos[self.ref_end_effectors, 2] - self.data.xpos[self.end_effectors, 2])),
        }
        thresholds = self.config["termination_thresholds"]
        return [name for name, error in errors.items() if error > thresholds[name]]

    def hand_contacts(self) -> tuple[bool, bool]:
        touching_geoms = set()
        for contact in self.data.contact:
            if contact.efc_address >= 0 and self.box_geom in contact.geom:
                touching_geoms.update(contact.geom)
        left, right = self.hand_geoms
        return left in touching_geoms, right in touching_geoms


def load_bundle(bundle: Path) -> tuple[dict, dict]:
    config = json.loads((bundle / "deployment.json").read_text())
    if config["version"] != BUNDLE_VERSION:
        raise ValueError(f"Unsupported deployment version {config['version']}")
    with np.load(bundle / "motion.npz", allow_pickle=False) as archive:
        motion = dict(archive)
    for key in ("body_quat_w", "ext_object_quat_w"):
        motion[key] = motion[key][..., [3, 0, 1, 2]]
    return config, motion


def evaluate(
    trial: TeacherPlayback,
    actor: Callable[[np.ndarray], np.ndarray],
    on_frame: Callable[[], bool] | None = None,
) -> dict:
    ref_pos = trial.motion["ext_object_pos_w"]
    start_height = ref_pos[0, 2]
    peak_lift = max_error = 0.0
    lift_streak = setdown_streak = 0
    lifted = set_down = False
    contact_frames = np.zeros(3, dtype=int)
    failures = []
    for frame, target in enumerate(ref_pos):
        trial.step(actor(trial.observation(frame)))
        position = trial.data.xpos[trial.box]
        error = float(np.linalg.norm(position - target))
        height = float(position[2] - start_height)
        ref_height = target[2] - start_height
        peak_lift = max(peak_lift, height)
        max_error = max(max_error, error)
        left, right = trial.hand_contacts()
        contact_frames += (left, right, left and right)
        if not lifted:
            lift_streak = lift_streak + 1 if height >= 0.25 and ref_height >= 0.25 and error <= 0.20 else 0
            lifted = lift_streak >= 25
        elif not set_down:
            speed = np.linalg.norm(trial.data.qvel[trial.box_qvel : trial.box_qvel + 3])
            setdown_now = ref_height <= 0.08 and abs(height) <= 0.08 and error <= 0.20 and speed <= 0.20
            setdown_streak = setdown_streak + 1 if setdown_now else 0
            set_down = setdown_streak >= 15
        failures = trial.failures(frame)
        if on_frame is not None and not on_frame():
            failures.append("viewer_closed")
        if failures:
            break
    completed = frame + 1 == len(ref_pos) and not failures
    final_pos_error = float(np.linalg.norm(trial.data.xpos[trial.box] - ref_pos[-1]))
    orientation = trial.data.xquat[trial.box]
    reference_orientation = trial.motion["ext_object_quat_w"][-1]
    quat_dot = np.dot(orientation, reference_orientation) / (
        np.linalg.norm(orientation) * np.linalg.norm(reference_orientation)
    )
    final_angle_error = float(2 * np.arccos(np.clip(abs(quat_dot), 0, 1)))
    return {
        "steps": frame + 1,
        "completed_clip": completed,
        "lifted": lifted,
        "set_down": set_down,
        "success": completed and lifted and set_down and final_pos_error < 0.15 and final_angle_error < 0.30,
        "peak_lift_m": peak_lift,
        "max_tracking_error_m": max_error,
        "final_position_error_m": final_pos_error,
        "final_angle_error_rad": final_angle_error,
        "contact_frames": dict(zip(("left", "right", "both"), contact_frames.tolist(), strict=True)),
        "failure_terms": failures,
        "domain_sample": trial.domain_sample,
    }


def play_bundle(
    bundle: Path,
    dr: bool,
    trials: int,
    seed: int,
    viewer: bool,
    box_xy_dr: float | None = None,
    box_xy_offset: tuple[float, float] | None = None,
) -> dict:
    import onnxruntime as ort

    config, motion = load_bundle(bundle)
    if box_xy_dr is not None:
        config["box_xy_dr_m"] = box_xy_dr
    if box_xy_offset is not None:
        config["box_xy_offset_m"] = box_xy_offset
        config["box_xy_dr_m"] = 0.0
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(bundle / "policy.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )

    def actor(observation: np.ndarray) -> np.ndarray:
        return session.run(["actions"], {"obs": observation[None]})[0][0]

    gui = None
    if viewer:
        from motrix_deploy_mujoco.viewer import MujocoGlfwViewer
        from motrix_env_core.config.scene import SystemCameraCfg

        positions = motion["ext_object_pos_w"]
        center = (positions[0] + positions[-1]) * 0.5
        camera = SystemCameraCfg(
            lookat=(float(center[0]), float(center[1]), 0.65), distance=3.4, azimuth=115, elevation=-15
        )
        gui = MujocoGlfwViewer(mj, camera)

    results = []
    for trial_seed in range(seed, seed + trials):
        trial = TeacherPlayback(bundle, config, motion, dr, trial_seed)
        try:
            if gui is not None:
                gui.open(trial.model, trial.data)
            start = time.monotonic()

            def draw() -> bool:
                try:
                    gui.sync()
                except KeyboardInterrupt:
                    return False
                time.sleep(max(0, start + trial.data.time - time.monotonic()))
                return True

            results.append({"seed": trial_seed, **evaluate(trial, actor, draw if gui is not None else None)})
        finally:
            if gui is not None:
                gui.close()
        if "viewer_closed" in results[-1]["failure_terms"]:
            break
    return {
        "task": config["task"],
        "source_checkpoint": config["source_checkpoint"],
        "dr": dr,
        "box_xy_dr_m": config.get("box_xy_dr_m", 0.0) if dr else 0.0,
        "reference_peak_lift_m": float(motion["ext_object_pos_w"][:, 2].max() - motion["ext_object_pos_w"][0, 2]),
        "completed_clip_rate": sum(result["completed_clip"] for result in results) / len(results),
        "carry_success_rate": sum(result["success"] for result in results) / len(results),
        "trials": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export the run's best playable policy and MuJoCo scene")
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--motion", required=True, type=Path)
    export.add_argument("--bundle", required=True, type=Path)
    play = commands.add_parser("play", help="Run the exported scene and policy in MuJoCo")
    play.add_argument("--bundle", required=True, type=Path)
    play.add_argument(
        "--dr", action=argparse.BooleanOptionalAction, default=False, help="Enable training perturbations"
    )
    play.add_argument("--trials", type=int, default=1)
    play.add_argument("--seed", type=int, default=0)
    box_xy = play.add_mutually_exclusive_group()
    box_xy.add_argument("--box-xy-dr", type=float, help="Override the bundle's initial box XY half-range in meters")
    box_xy.add_argument("--box-xy-offset", nargs=2, type=float, metavar=("X", "Y"),
                        help="Fix the initial box offset from the reference in world X/Y meters")
    play.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer")
    play.add_argument("--report", type=Path, help="Write a copy of the JSON evaluation report")
    args = parser.parse_args()
    if args.command == "export":
        export_bundle(args.run_dir, args.motion, args.bundle)
    else:
        if args.trials < 1 or args.seed < 0:
            parser.error("--trials must be positive and --seed must be nonnegative")
        if args.box_xy_dr is not None and (not isfinite(args.box_xy_dr) or args.box_xy_dr < 0):
            parser.error("--box-xy-dr must be a finite nonnegative half-range in meters")
        if args.box_xy_offset is not None and not all(map(isfinite, args.box_xy_offset)):
            parser.error("--box-xy-offset requires two finite offsets in meters")
        offset = tuple(args.box_xy_offset) if args.box_xy_offset is not None else None
        result = play_bundle(args.bundle, args.dr, args.trials, args.seed, args.viewer, args.box_xy_dr, offset)
        output = json.dumps(result, indent=2) + "\n"
        if args.report is not None:
            args.report.write_text(output)
        print(output, end="")


if __name__ == "__main__":
    main()
