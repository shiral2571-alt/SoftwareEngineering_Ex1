"""Turning a list of projected images into an mp4 video."""
from __future__ import annotations

import numpy as np

MACRO_BLOCK = 16


class VideoWriter:
    """Writes frames of possibly different sizes and any integer dtype to mp4.

    - Intensities are normalized with one global window over all frames, so
      the same voxel value has the same gray level in every frame. The upper
      bound is the `clip_percentile` of all pixels: a few very bright voxels
      would otherwise squeeze the rest of the image into dark grays.
    - Every frame is padded (centered) to a common size rounded up to a
      multiple of 16, as required by the H.264 encoder.
    """

    def __init__(self, fps: float = 10, clip_percentile: float = 99.5):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 < clip_percentile <= 100:
            raise ValueError("clip_percentile must be in (0, 100]")
        self.fps = fps
        self.clip_percentile = clip_percentile

    def write(self, frames: list[np.ndarray], path: str) -> None:
        import imageio.v2 as imageio
        images = self.prepare(frames, self.clip_percentile)
        imageio.mimwrite(path, images, fps=self.fps, macro_block_size=1)

    @staticmethod
    def prepare(frames: list[np.ndarray], clip_percentile: float = 100) -> list[np.ndarray]:
        """Normalize to uint8 and pad to a common size (no I/O, testable)."""
        if not frames:
            raise ValueError("No frames to write")

        low = min(float(frame.min()) for frame in frames)
        if clip_percentile < 100:
            all_pixels = np.concatenate([frame.ravel() for frame in frames])
            high = float(np.percentile(all_pixels, clip_percentile))
        else:
            high = max(float(frame.max()) for frame in frames)
        height = max(frame.shape[0] for frame in frames)
        width = max(frame.shape[1] for frame in frames)
        height += (-height) % MACRO_BLOCK
        width += (-width) % MACRO_BLOCK

        images = []
        for frame in frames:
            if high > low:
                normalized = np.clip((frame.astype(np.float64) - low) / (high - low), 0, 1)
            else:
                normalized = np.zeros(frame.shape)
            image = np.rint(normalized * 255).astype(np.uint8)
            images.append(pad_to_size(image, height, width))
        return images


def pad_to_size(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center `image` inside a zero canvas of (height, width)."""
    pad_h = height - image.shape[0]
    pad_w = width - image.shape[1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size is smaller than the image")
    top, left = pad_h // 2, pad_w // 2
    return np.pad(image, ((top, pad_h - top), (left, pad_w - left)), mode="constant")
