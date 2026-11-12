"""Validate training targets against the existing PC2 processor's actual method."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import motrix_envs  # noqa: F401
from motrix_deploy.contracts import JointServoCommand, float32_array
from motrix_env_core import registry
from motrix_envs.deploy.g1_wbt import build_g1_wbt_profile
from motrix_envs.locomotion.wbt.g1.largebox import RobotOnlyCarryEnv

ROOT = Path(__file__).resolve().parents[2]


def main():
    profile = build_g1_wbt_profile("g1-29dof-wbt-largebox")
    cfg = registry.make_env_config("g1-29dof-wbt-largebox")
    env = RobotOnlyCarryEnv(cfg, num_envs=8, seed=1)
    try:
        env.init_state()
        term = env.action_terms["joint_position"]
        robot, spec = profile.robot, profile.task
        assert tuple(env.model.bodies["robot"].joint_names) == robot.joint_names
        np.testing.assert_allclose(term.state.action_scales, spec.action_scale, atol=1e-7)
        np.testing.assert_allclose(term.state.default_angles, robot.default_joint_position, atol=1e-7)
        np.testing.assert_allclose(term.state.joint_lower, robot.position_lower, atol=1e-6)
        np.testing.assert_allclose(term.state.joint_upper, robot.position_upper, atol=1e-6)
        pc2 = ROOT / "deployments/g1_pc2_compat/src/motrix_deploy_pc2/wbt_box.py"
        tree = ast.parse(pc2.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "G1WbtPolicyProcessor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_action")
        namespace = dict(np=np, float32_array=float32_array, JointServoCommand=JointServoCommand, FloatArray=np.ndarray)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(pc2), "exec"), namespace)
        processor = SimpleNamespace(
            robot=robot,
            spec=spec,
            _default_joint_position=robot.default_joint_position,
            _action_scale=np.asarray(spec.action_scale, np.float32),
            _kp=np.asarray(spec.kp, np.float32),
            _kd=np.asarray(spec.kd, np.float32),
        )
        rng = np.random.default_rng(1)
        max_error = 0.0
        for raw in [
            np.zeros(29, np.float32),
            np.full(29, 100, np.float32),
            np.full(29, -100, np.float32),
            *rng.uniform(-30, 30, (100, 29)).astype(np.float32),
        ]:
            expected = namespace["process_action"](processor, raw).joint_position
            actual = term._process(raw[None])[0]
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            np.testing.assert_array_equal(processor._previous_action, raw)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        motion = env.command_terms["motion"].clip
        lo, hi = term._target_limits
        violations = (motion.joint_pos < lo) | (motion.joint_pos > hi)
        report = dict(
            frames=len(motion.joint_pos),
            fps=50,
            clip_fraction=spec.action_safety_clip_fraction,
            action_parity_samples=103,
            max_abs_error=max_error,
            frames_outside_clip=int(np.any(violations, axis=1).sum()),
            joints_outside_clip={
                name: int(violations[:, i].sum())
                for i, name in enumerate(robot.joint_names)
                if np.any(violations[:, i])
            },
        )
        ground = env.sim._model.get_geom("floor")
        friction = ground.get_friction_override(env.sim._data).copy()
        assert np.ptp(friction[:, 0]) > 0
        assert np.all((friction[:, 0] >= 0.6) & (friction[:, 0] <= 1.6))
        np.testing.assert_allclose(friction[:, 1:], np.tile(env.model.others["ground_action_friction"][1:], (8, 1)))
        for ids, sim_ids in [(np.arange(4), None), (np.empty(0, dtype=np.int64), np.arange(2))]:
            env.reset(ids, sim_ids)
            np.testing.assert_array_equal(ground.get_friction_override(env.sim._data), friction)
        env._refresh_sim_reads()
        before = {
            key: env.sim_data[source].copy()
            for key, source in [("linear", "root_lin_vel"), ("angular", "root_ang_vel")]
        }
        # Pushes are independently disabled while ground DR remains enabled.
        env._push_remaining[:] = 0.0
        countdown = env._push_remaining.copy()
        buffers = {key: env._push_writes.buffer(key).copy() for key in ("linear", "angular")}
        env.physics_step()
        np.testing.assert_array_equal(env._push_remaining, countdown)
        for key, values in buffers.items():
            np.testing.assert_array_equal(env._push_writes.buffer(key), values)
        report["push_disabled_with_ground_dr_enabled"] = True
        # Opt in only inside this diagnostic to retain validation of the push implementation.
        env.cfg.randomization.push_enabled = True
        env._refresh_sim_reads()
        before = {
            key: env.sim_data[source].copy()
            for key, source in [("linear", "root_lin_vel"), ("angular", "root_ang_vel")]
        }
        env._push_remaining[:] = 2.0
        env._push_remaining[0] = 0.0
        env.physics_step()
        kicks = {}
        for key, amplitude in [("linear", env._push_linear), ("angular", env._push_angular)]:
            delta = env._push_writes.buffer(key)[0, 0] - before[key][0]
            assert np.all(np.abs(delta) <= amplitude + 1e-6)
            assert np.any(delta != 0)
            kicks[key] = delta.tolist()
        assert 1.0 <= env._push_remaining[0] <= 3.0
        np.testing.assert_allclose(env._push_remaining[1:], 1.98)
        report.update(
            ground_friction=friction[:, 0].tolist(),
            friction_reset_persistence=True,
            forced_push_delta=kicks,
            push_interval_s=[1.0, 3.0],
        )
        out = ROOT / "artifacts/g1-robot-only-clip-validation/preflight.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        del env


if __name__ == "__main__":
    main()
