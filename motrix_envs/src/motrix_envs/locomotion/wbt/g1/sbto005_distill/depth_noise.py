# Copyright Motphys Technology Co., Ltd. 2026
# SPDX-License-Identifier: Apache-2.0

"""Student-side RealSense depth perturbations with lane-local episode history."""

import torch

from motrix_envs.locomotion.wbt.g1.sbto005_distill.camera import FAR, HEIGHT, NEAR, WIDTH, student_depth
from motrix_envs.locomotion.wbt.g1.sbto005_distill.visibility import MIN_BOX_PIXELS

RAMP_STEPS = 120_000
GAUSSIAN_STD = 0.004
SALT_PROB = 0.01
RECT_COUNT = 2
RECT_SIZE = (0.1, 0.3)
MAX_LAG = 2
HOLD_PROB = 0.05
MASK_RETRIES = 8


class DepthNoise:
    def __init__(self, num_envs: int, device: torch.device, seed: int):
        self.device = device
        self._rng = torch.Generator(device=device).manual_seed(seed)
        self._height = torch.arange(HEIGHT, device=device).view(1, HEIGHT, 1)
        self._width = torch.arange(WIDTH, device=device).view(1, 1, WIDTH)
        self._rows = torch.arange(num_envs, device=device)
        shape = (num_envs, 1, HEIGHT, WIDTH)
        self._history = torch.ones((MAX_LAG + 1, *shape), device=device)
        self._delivered = torch.ones(shape, device=device)
        self._slot = 0
        self.first_frames = 0
        self.first_frame_passes = 0
        self.first_frame_fallbacks = 0
        self.first_box_pixels = torch.zeros(num_envs, dtype=torch.long)

    def _uniform(self, *shape):
        return torch.rand(shape, generator=self._rng, device=self.device)

    def _dropout(self, image, strength):
        n = image.shape[0]
        drop = self._uniform(n, HEIGHT, WIDTH) < SALT_PROB * strength
        counts = (self._uniform(n, 1, 1) * (RECT_COUNT + 1) * strength).floor()
        lo, hi = RECT_SIZE
        for index in range(RECT_COUNT):
            y, x, h, w = self._uniform(4, n, 1, 1).unbind(0)
            rect = (self._height - y * HEIGHT).abs() < (lo + h * (hi - lo)) * HEIGHT / 2
            rect = rect & ((self._width - x * WIDTH).abs() < (lo + w * (hi - lo)) * WIDTH / 2)
            drop |= (counts > index) & rect
        return image.masked_fill(drop.unsqueeze(1), 1.0)

    def frame(self, raw_depth, fresh, box_pixel_mask, strength: float):
        """Return the exact fp16-quantized image stored in the DAgger replay."""
        clean = student_depth(raw_depth.to(self.device))
        fresh_dev = fresh.to(self.device)
        if strength > 0.0:
            gaussian = torch.randn(clean.shape, generator=self._rng, device=self.device) * (GAUSSIAN_STD * strength)
            # Keep the invalid-depth sentinel distinct, including after replay quantization.
            noisy = torch.where(clean < 1, (clean + gaussian).clamp_(NEAR / FAR, 1.0 - 2**-11), clean)
            image = self._dropout(noisy, strength)
        else:
            noisy = clean
            image = clean

        if fresh.numel():
            mask = box_pixel_mask[fresh].to(self.device)
            pixels = ((image[fresh_dev].half() < 1) & mask).flatten(1).sum(1)
            missing = torch.nonzero(pixels < MIN_BOX_PIXELS).flatten()
            for _ in range(MASK_RETRIES):
                if not missing.numel():
                    break
                rows = fresh_dev[missing]
                images = self._dropout(noisy[rows], strength)
                image[rows] = images
                pixels[missing] = ((images.half() < 1) & mask[missing]).flatten(1).sum(1)
                missing = missing[pixels[missing] < MIN_BOX_PIXELS]
            if missing.numel():
                self.first_frame_fallbacks += len(missing)
                rows = fresh_dev[missing]
                image[rows] = noisy[rows]
                pixels[missing] = ((image[rows].half() < 1) & mask[missing]).flatten(1).sum(1)
            counts = pixels.cpu()
            passed = int((counts >= MIN_BOX_PIXELS).sum())
            if passed != len(fresh):
                raise RuntimeError(f"First student depth frame is missing the required box pixels: {counts.tolist()}")
            self.first_frames += len(fresh)
            self.first_frame_passes += passed
            self.first_box_pixels[fresh] = counts

        self._history[self._slot].copy_(image)
        self._history[:, fresh_dev] = image[fresh_dev]
        self._delivered[fresh_dev] = image[fresh_dev]
        if strength > 0.0:
            lag = torch.randint(MAX_LAG + 1, self._rows.shape, generator=self._rng, device=self.device)
            lag *= self._uniform(len(lag)) < strength
            delayed = self._history[(self._slot - lag) % (MAX_LAG + 1), self._rows]
            hold = self._uniform(len(lag)) < HOLD_PROB * strength
            self._delivered.copy_(torch.where(hold[:, None, None, None], self._delivered, delayed))
        else:
            self._delivered.copy_(image)
        self._slot = (self._slot + 1) % (MAX_LAG + 1)
        return self._delivered.half()
