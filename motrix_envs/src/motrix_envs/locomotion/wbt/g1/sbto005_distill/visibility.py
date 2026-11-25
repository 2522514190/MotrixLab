# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Accept only reset placements visible in the student's depth camera."""

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.sim_reset import ResetTerm
from motrix_env_core.math.quaternion import rotate_inverse, rotate_vector
from motrix_env_core.sim.read import SitePositionQuery, SiteQuaternionQuery
from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import DEPTH, FAR, HEIGHT, NEAR, SITE, WIDTH, projection
from motrix_envs.locomotion.wbt.g1.sbto005_teacher import BOX_HALF_EXTENTS, Sbto005TeacherCfg, Sbto005TeacherEnv
from motrix_envs.locomotion.wbt.mdp.sbto005_teacher import BoxResetCfg, reset_box

XY_LOW = (-0.10, -0.10)
XY_HIGH = (0.10, 0.02)
YAW_DEG = 30.0
MIN_BOX_PIXELS = 20
XY_RETRIES = 4
POSE_RETRIES = 3


def reset_visible_box(ctx: ManagerContext, env_ids: torch.Tensor, writes: Map[torch.Tensor], yaw: float) -> None:
    reset_box(ctx, env_ids, writes, 0.0, yaw)
    x, y = ctx.rand.uniform((env_ids.numel(), 2)).unbind(dim=-1)
    frame_zero = ctx.commands["motion"].steps[env_ids, 0] == 0
    writes["position"][:, 0, 0].add_(frame_zero * (XY_LOW[0] + (XY_HIGH[0] - XY_LOW[0]) * x))
    writes["position"][:, 0, 1].add_(frame_zero * (XY_LOW[1] + (XY_HIGH[1] - XY_LOW[1]) * y))


@configclass(kw_only=True)
class VisibleBoxResetCfg(BoxResetCfg):
    def __call__(self, ctx: BuildContext) -> ResetTerm:
        return ResetTerm(reset_visible_box, self.initial_yaw_noise, writes=super().__call__(ctx).writes)


def configure_visible_box(cfg: Sbto005TeacherCfg) -> None:
    cfg.sim_reset.object_state = VisibleBoxResetCfg(
        body=cfg.sim_reset.object_state.body, initial_yaw_noise=YAW_DEG * (torch.pi / 180)
    )
    site = cfg.scene.objs.robot.resolve_name(SITE)
    cfg.queries.data.update(
        visibility_camera_pos=SitePositionQuery(site=site), visibility_camera_quat=SiteQuaternionQuery(site=site)
    )


class VisibleBoxEnv(Sbto005TeacherEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lens = projection()
        y, x = torch.meshgrid(torch.arange(HEIGHT) + 0.5, torch.arange(WIDTH) + 0.5, indexing="ij")
        self._rays = torch.stack(((x - lens.cx) / lens.fx, (y - lens.cy) / lens.fy, torch.ones_like(x)), -1).flatten(
            0, 1
        )
        self._box_size = torch.tensor(BOX_HALF_EXTENTS)
        self._xy_low = torch.tensor(XY_LOW)
        self._xy_span = torch.tensor(XY_HIGH) - self._xy_low
        self._position_rng = torch.Generator().manual_seed(self._build_seed + 71219)
        self.box_pixel_mask = torch.zeros((self.num_envs, 1, HEIGHT, WIDTH), dtype=torch.bool)
        self.box_first_draw = torch.zeros(self.num_envs, dtype=torch.bool)
        self.visibility_counts = dict(resets=0, first_draw_passes=0, xy_retries=0, pose_retries=0)
        self.visibility_histogram = torch.zeros((10, 6), dtype=torch.long)

    def box_pixels(self, rows: torch.Tensor) -> torch.Tensor:
        data = self.sim_data
        box_quat = data["object_quat"][rows]
        ray_origin = rotate_inverse(box_quat, data["visibility_camera_pos"][rows] - data["object_pos"][rows])
        ray_world = rotate_vector(data["visibility_camera_quat"][rows, None], self._rays)
        ray = rotate_inverse(box_quat[:, None], ray_world)
        # Rays have optical z = 1, so the slab parameter is metric z-depth.
        divisor = torch.where(ray.abs() > 1e-8, ray, 1e-8)
        low = (-self._box_size - ray_origin[:, None]) / divisor
        high = (self._box_size - ray_origin[:, None]) / divisor
        z_in = torch.minimum(low, high).amax(dim=-1)
        z_out = torch.maximum(low, high).amin(dim=-1)
        actual = data[DEPTH][rows].flatten(1)
        mask = (z_out >= z_in) & (z_in >= NEAR) & (actual < FAR) & ((actual - z_in).abs() <= 0.003)
        return mask.reshape(-1, 1, HEIGHT, WIDTH)

    def _unseen(self, rows: torch.Tensor) -> torch.Tensor:
        mask = self.box_pixels(rows)
        self.box_pixel_mask[rows] = mask
        return rows[mask.flatten(1).sum(dim=1) < MIN_BOX_PIXELS]

    def _refresh_box_reset(self, rows: torch.Tensor) -> None:
        # The native reset refreshes camera pixels. A pose-only FK write leaves the cached image stale.
        self._sim_reset_runtime.program.execute(rows)
        if self._domain_writes is not None:
            self._domain_writes.execute(rows)
        self._compiled_manager_program.read_plan.read(self, rows)

    def _reset_sim_rows(self, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        super()._reset_sim_rows(env_ids, episode_reset)
        rows = env_ids[episode_reset]
        if not rows.numel():
            return

        reference = self.command_terms["motion"].clip.extensions["object_pos_w"].data[0, :2]
        reset = self._sim_reset_runtime.program
        positions = reset.buffer("object_state.position")
        rotations = reset.buffer("object_state.rotation")
        initial_rotation = rotations.clone()
        pending = self._unseen(rows)
        self.box_first_draw[rows] = True
        self.box_first_draw[pending] = False
        self.visibility_counts["resets"] += len(rows)
        self.visibility_counts["first_draw_passes"] += len(rows) - len(pending)

        for attempt in range(POSE_RETRIES + 1):
            if not pending.numel():
                break
            if attempt:
                self.visibility_counts["pose_retries"] += len(pending)
                super()._reset_sim_rows(pending, torch.ones_like(pending, dtype=torch.bool))
                rotations[pending] = initial_rotation[pending]
                self._refresh_box_reset(pending)
                pending = self._unseen(pending)
            for _ in range(XY_RETRIES):
                if not pending.numel():
                    break
                self.visibility_counts["xy_retries"] += len(pending)
                unit = torch.rand((len(pending), 2), generator=self._position_rng)
                positions[pending, 0, :2] = reference + self._xy_low + self._xy_span * unit
                self._refresh_box_reset(pending)
                pending = self._unseen(pending)

        if pending.numel():
            pixels = self.box_pixel_mask[pending].flatten(1).sum(dim=1)
            raise RuntimeError(
                f"Box visibility reset exhausted for lanes {pending.tolist()} ({pixels.tolist()} pixels)."
            )

        xy = self.sim_data["object_pos"][rows, :2] - reference
        bins = ((xy - self._xy_low) / 0.02).long().clamp_min_(0)
        indices = bins[:, 0].clamp_max(9) * 6 + bins[:, 1].clamp_max(5)
        self.visibility_histogram.add_(torch.bincount(indices, minlength=60).view(10, 6))
