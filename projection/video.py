"""Turning a list of projected images into an mp4 video."""
from __future__ import annotations

import numpy as np

MACRO_BLOCK = 16


class VideoWriter:
    """Writes frames of possibly different sizes and any integer dtype to mp4.

    - Intensities are normalized with one global min/max over all frames, so
      the same voxel value has the same gray level in every frame.
    - Every frame is padded (centered) to a common size rounded up to a
      multiple of 16, as required by the H.264 encoder.
    """

    def __init__(self, fps: float = 10):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps

    def write(self, frames: list[np.ndarray], path: str) -> None:
        import imageio.v2 as imageio
        images = self.prepare(frames)
        imageio.mimwrite(path, images, fps=self.fps, macro_block_size=1)

    @staticmethod
    def prepare(frames: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize to uint8 and pad to a common size (no I/O, testable)."""
        if not frames:
            raise ValueError("No frames to write")

        low = min(float(frame.min()) for frame in frames)
        high = max(float(frame.max()) for frame in frames)
        height = max(frame.shape[0] for frame in frames)
        width = max(frame.shape[1] for frame in frames)
        height += (-height) % MACRO_BLOCK
        width += (-width) % MACRO_BLOCK

        images = []
        for frame in frames:
            if high > low:
                normalized = (frame.astype(np.float64) - low) / (high - low)
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
