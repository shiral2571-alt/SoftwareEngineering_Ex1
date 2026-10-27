"""Save a strip of evenly spaced frames from each result video (results/*.png)."""
import glob
import os

import imageio.v2 as imageio
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

for video in sorted(glob.glob(os.path.join(RESULTS, "*.mp4"))):
    with imageio.get_reader(video) as reader:
        frames = [frame[..., 0] for frame in reader]
    picks = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    strip = np.hstack([frames[i] for i in picks])
    out = video[:-4] + "_preview.png"
    imageio.imwrite(out, strip)
    print(out, strip.shape)
