# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Verify and benchmark the SBTO005 student camera with a frozen teacher."""

import argparse
import json
import resource
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np
import torch
from evaluate_sbto005_teacher import EvaluationEnv
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap
from PIL import Image, ImageDraw
from torch import nn

from motrix_env_core import registry
from motrix_env_core.math.quaternion import inverse, mul, rotate_inverse, rotate_vector
from motrix_env_core.sim.read import LinkPositionQuery, LinkQuaternionQuery, SitePositionQuery, SiteQuaternionQuery
from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import (
    DEPTH,
    FAR,
    HEIGHT,
    MOUNT_POSITION,
    NEAR,
    SITE,
    WIDTH,
    aligned_ray_pixels,
    attach_depth_camera,
    camera_sensor,
    isolate_head_for_rays,
    optical_rotation,
    projection,
    resample_left_ir_depth,
    student_depth,
)
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import TASK, Sbto005TeacherEnv
from motrix_rl import runner, runs


def make_env(policy, count, mode, seed, *, diagnostic=False):
    run_dir, metadata = runs.find_metadata_for_policy(policy)
    run = runs.open_run_context(run_dir, metadata)
    trainer = runner.create_run_handle(run, cfg_override={"device": "cpu"}, play_num_envs=count, seed=seed).trainer
    torch.manual_seed(seed)
    cfg = registry.make_env_config(TASK, mode="play")
    cfg.commands.motion.adaptive_sampling_enabled = False
    cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    if mode == "unfiltered":
        attach_depth_camera(cfg, near=0.0 if diagnostic else NEAR)
    with TemporaryDirectory(prefix="sbto-camera-") as tmp:
        if mode in ("filtered", "cnn"):
            model = Path(tmp) / "g1.xml"
            isolate_head_for_rays(Path(cfg.scene.objs.robot.model.file), model)
            attach_depth_camera(cfg, robot_mjcf=model, near=NEAR)
        if diagnostic:
            site = cfg.scene.objs.robot.resolve_name(SITE)
            cfg.queries.data.update(
                camera_pos=SitePositionQuery(site=site),
                camera_quat=SiteQuaternionQuery(site=site),
                torso_pos=LinkPositionQuery(link="torso_link"),
                torso_quat=LinkQuaternionQuery(link="torso_link"),
            )
        env_cls = EvaluationEnv if diagnostic else Sbto005TeacherEnv
        inner = env_cls(cfg, num_envs=count, backend=metadata.sim, seed=seed, headless=True)
    env = FastSacEnvWrap(inner, torch.device("cpu"))
    agent = trainer._make_agent(env)
    checkpoint = torch.load(policy, map_location="cpu", weights_only=True)
    assert checkpoint["global_step"] == 410000
    agent.load_state_dict(checkpoint, load_optimizers=False)
    agent.actor.eval()
    agent.obs_normalizer.eval()
    return env, agent


def save_sheet(path, images, frames, title):
    sheet = Image.new("RGB", (WIDTH * 4 * len(frames), HEIGHT * 4 + 25))
    draw = ImageDraw.Draw(sheet)
    for column, (frame, depth) in enumerate(zip(frames, images, strict=True)):
        valid = np.isfinite(depth) & (depth > 0)
        gray = np.where(valid, np.clip(depth / FAR, 0, 1), 0)
        rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        rgb[~valid] = (180, 0, 120)
        x = column * WIDTH * 4
        sheet.paste(Image.fromarray(rgb).resize((WIDTH * 4, HEIGHT * 4), Image.Resampling.NEAREST), (x, 25))
        draw.text((x + 4, 4), f"{title} {frame}", fill="white")
    sheet.save(path)


@torch.inference_mode()
def snapshots(args):
    frames = [0, 90, 123, 160, 240, 335, 450, 495]
    traces = {}
    validity = {}
    for mode in ("unfiltered", "filtered"):
        env, agent = make_env(args.policy, args.envs, mode, args.seed, diagnostic=True)
        images, trace = [], []
        obs, _ = env.reset()
        sim = env.env.sim_data
        offset = rotate_inverse(sim["torso_quat"], sim["camera_pos"] - sim["torso_pos"])
        rotation = mul(inverse(sim["torso_quat"]), sim["camera_quat"])
        torch.testing.assert_close(offset, torch.tensor(MOUNT_POSITION).expand_as(offset), atol=1e-6, rtol=0)
        axis = torch.eye(3)
        axes = rotate_vector(rotation[:1].expand(3, 4), axis)
        expected_axes = rotate_vector(torch.tensor(optical_rotation()).expand(3, 4), axis)
        torch.testing.assert_close(axes, expected_axes, atol=1e-6, rtol=0)
        for frame in range(frames[-1] + 1):
            if frame in frames:
                images.append(env.env.sim_data[DEPTH].clone())
            state = env.env.sim_data
            trace.append(
                torch.cat(
                    [
                        state[key]
                        for key in ("object_pos", "object_quat", "object_lin_vel", "robot_dof_pos", "robot_dof_vel")
                    ],
                    dim=-1,
                )
            )
            if frame != frames[-1]:
                obs = env.step(agent.act(obs, deterministic=True)).obs
        traces[mode] = torch.stack(trace)
        depth = torch.stack(images)
        prepared = torch.stack([student_depth(image) for image in images])
        invalid = ~torch.isfinite(depth) | (depth <= 0)
        assert prepared.shape == (len(frames), args.envs, 1, HEIGHT, WIDTH)
        assert torch.isfinite(prepared).all() and prepared.min() >= NEAR / FAR - 1e-7 and prepared.max() <= 1
        validity[mode] = (~invalid).float().mean(dim=(1, 2, 3)).tolist()
        np.savez_compressed(args.out / f"{mode}.npz", frames=frames, depth=depth, invalid=invalid, student=prepared)
        save_sheet(args.out / f"{mode}.png", depth[:, 0].numpy(), frames, mode)
        save_sheet(args.out / f"{mode}-student.png", prepared[:, 0, 0].numpy() * FAR, frames, "student")
        env.close()
    torch.testing.assert_close(traces["unfiltered"], traces["filtered"], rtol=0, atol=1e-6)
    states = traces["filtered"]
    result = {
        "frames": frames,
        "environments": args.envs,
        "valid_pixel_fraction_by_frame": validity,
        "mount_position_in_torso_m": offset[0].tolist(),
        "optical_right_down_forward_in_torso": axes.tolist(),
        "state_max_abs_difference": float((states - traces["unfiltered"]).abs().max()),
        "peak_box_lift_m": (states[:, :, 2].amax(dim=0) - states[0, :, 2]).tolist(),
    }
    (args.out / "snapshots.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


def geometry(args):
    from motrix_env_core.config import SimCfg, configclass
    from motrix_env_core.config.scene.base import SceneCfg, SceneSensorsCfg
    from motrix_env_core.config.scene.sensor import RaycastCameraSensorCfg
    from motrix_env_core.sim.read import RaycastCameraImageQuery
    from motrix_env_core.sim.write import JointPositionWrite, JointVelocityWrite
    from motrix_env_motrixsim.runtime import MotrixSimBackend

    x, y = aligned_ray_pixels()
    np.testing.assert_array_equal(x, 109 + 10 * np.arange(WIDTH))
    np.testing.assert_array_equal(y, 5 + 10 * np.arange(HEIGHT))
    index_image = np.arange(480 * 848).reshape(480, 848)
    np.testing.assert_array_equal(resample_left_ir_depth(index_image), y[:, None] * 848 + x)
    raw = torch.tensor([[[float("nan"), 0.0, -1.0, 0.1, 1.0, 3.0, float("inf")]]])
    torch.testing.assert_close(student_depth(raw).flatten(), torch.tensor([1.0, 1.0, 1.0, 0.168 / 2.5, 0.4, 1.0, 1.0]))

    @configclass
    class Sensors(SceneSensorsCfg):
        depth: RaycastCameraSensorCfg = camera_sensor(site="optical", filtered=False)

    p = projection()
    rays_x = (np.arange(WIDTH) + 0.5 - p.cx) / p.fx
    measured = {}
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.xml"
        for tilt in (0.0, 0.22):
            path.write_text(
                '<mujoco><option gravity="0 0 0"/><worldbody>'
                f'<geom type="box" size="4 4 .05" pos="0 0 1.4" euler="0 {np.rad2deg(tilt)} 0"/>'
                '<body name="rig"><joint name="slide" type="slide" axis="0 0 1"/>'
                '<geom type="sphere" size=".01" pos="0 0 -.1" mass="1"/>'
                '<site name="optical"/></body></worldbody></mujoco>'
            )
            sim = MotrixSimBackend(SceneCfg(file=path, sensors=Sensors()), SimCfg(dt=0.005), 2)
            query = sim.compile_reads({"depth": RaycastCameraImageQuery(sensor="depth")})
            reset = sim.compile_writes(
                {"q": JointPositionWrite(joints=("slide",))}, reset=True, forward_kinematics=True
            )
            reset.execute()
            query.execute()
            target = (1.4 * np.cos(tilt) - 0.05) / (np.cos(tilt) + np.sin(tilt) * rays_x)
            expected = np.broadcast_to(target, (2, HEIGHT, WIDTH))
            np.testing.assert_allclose(query["depth"], expected, atol=2e-5)
            measured[str(tilt)] = float(np.max(np.abs(query["depth"].numpy() - expected)))

        velocity = sim.compile_writes({"v": JointVelocityWrite(joints=("slide",))})
        velocity.buffer("v")[:] = 1
        velocity.execute()
        changes = []
        previous = query["depth"].clone()
        for substep in range(1, 21):
            sim.step(1)
            query.execute()
            if not torch.equal(query["depth"], previous):
                changes.append(substep)
            previous.copy_(query["depth"])
        assert len(changes) >= 4 and set(np.diff(changes)) == {4}, changes
        before_reset = query["depth"].clone()
        reset.buffer("q")[:] = 0.3
        reset.execute(torch.tensor([1]))
        query.execute()
        torch.testing.assert_close(query["depth"][0], before_reset[0], atol=0, rtol=0)
        np.testing.assert_allclose(
            query["depth"][1],
            np.broadcast_to(
                ((1.4 - 0.3) * np.cos(tilt) - 0.05) / (np.cos(tilt) + np.sin(tilt) * rays_x), (HEIGHT, WIDTH)
            ),
            atol=2e-5,
        )
        boxes = {}
        images = []
        placements = [
            ("left", -0.18, 0.85, 0.0),
            ("center", 0.0, 0.85, 0.0),
            ("right", 0.18, 0.85, 0.0),
            ("far", 0.0, 1.35, 0.0),
            ("turned", 0.0, 0.85, 30.0),
        ]
        for label, x_pos, z_pos, degrees in placements:
            # Rotate about optical Y to mimic yaw for a level, forward-facing optical rig.
            path.write_text(
                '<mujoco><worldbody><site name="optical"/>'
                f'<geom type="box" size=".125 .14 .065" pos="{x_pos} 0 {z_pos}" '
                f'euler="0 {degrees} 0"/></worldbody></mujoco>'
            )
            backend = MotrixSimBackend(SceneCfg(file=path, sensors=Sensors()), SimCfg(dt=0.005), 1)
            image_read = backend.compile_reads({"depth": RaycastCameraImageQuery(sensor="depth")})
            backend.step(1)
            image_read.execute()
            image = image_read["depth"][0].numpy().copy()
            mask = np.isfinite(image) & (image > 0)
            cols = np.where(mask)[1]
            assert cols.size
            boxes[label] = dict(col=float(cols.mean()), area=int(mask.sum()), depth=float(np.median(image[mask])))
            images.append(image)
        assert boxes["left"]["col"] < boxes["center"]["col"] < boxes["right"]["col"]
        assert boxes["far"]["area"] < boxes["center"]["area"]
        assert boxes["far"]["depth"] > boxes["center"]["depth"]
        assert not np.array_equal(images[1], images[-1])
    save_sheet(args.out / "controlled-box.png", images, [case[0] for case in placements], "box")
    result = dict(
        plane_max_error_m=measured,
        refresh_substeps=changes,
        physics_dt=0.005,
        partial_reset="passed",
        controlled_box=boxes,
    )
    (args.out / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


def benchmark(args):
    env, agent = make_env(args.policy, args.envs, args.mode, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(384, 29),
        ).to(device)
        if args.mode == "cnn"
        else None
    )
    optimizer = torch.optim.Adam(model.parameters()) if model is not None else None
    obs, _ = env.reset()
    timings = dict(teacher=0.0, physics=0.0, read=0.0, preprocess=0.0, transfer=0.0, cnn=0.0)
    inner = env.env

    def timed(key, operation):
        def call(*pos, **kw):
            start = perf_counter()
            result = operation(*pos, **kw)
            timings[key] += perf_counter() - start
            return result

        return call

    inner.sim.step = timed("physics", inner.sim.step)
    original_refresh = inner._refresh_sim_reads
    inner._refresh_sim_reads = timed("read", original_refresh)
    latencies = []
    for index in range(args.steps + args.warmup):
        if index == args.warmup:
            timings = dict.fromkeys(timings, 0.0)
            start, cpu_start = perf_counter(), process_time()
            if model is not None and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
        step_start = perf_counter()
        with torch.no_grad():
            action = timed("teacher", agent.act)(obs, deterministic=True)
        if model is not None:
            depth = timed("preprocess", student_depth)(inner.sim_data[DEPTH])

            def transfer():
                return depth.to(device), action.to(device)

            pixels, labels = timed("transfer", transfer)()
            stage_start = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(pixels), labels)
            if index == 0 or index == args.steps + args.warmup - 1:
                assert torch.isfinite(loss).item()
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["cnn"] += perf_counter() - stage_start
        with torch.no_grad():
            obs = env.step(action).obs
        if index >= args.warmup:
            latencies.append(perf_counter() - step_start)
    wall = perf_counter() - start
    result = {
        "envs": args.envs,
        "mode": args.mode,
        "steps": args.steps,
        "env_steps_per_s": args.envs * args.steps / wall,
        "p95_control_ms": float(np.percentile(latencies, 95) * 1000),
        "cpu_cores_used": (process_time() - cpu_start) / wall,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_torch_gpu_mib": torch.cuda.max_memory_allocated() / 2**20
        if model is not None and device.type == "cuda"
        else 0,
        "ms_per_batch": {key: value * 1000 / args.steps for key, value in timings.items()},
    }
    env.close()
    (args.out / f"benchmark-{args.envs}-{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["geometry", "snapshots", "off", "filtered", "cnn"])
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.mode == "geometry":
        geometry(args)
        return
    if args.policy is None:
        parser.error("--policy is required outside geometry mode")
    if args.mode == "snapshots":
        snapshots(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
