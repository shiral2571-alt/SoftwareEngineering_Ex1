"""Demo: project an NRRD volume onto every plane in a CSV and save an mp4.

Usage:
    python Ex0.py file.nrrd file.csv file.mp4 [--fps N]
"""
import argparse
import sys
import time

import numpy as np

from projection import FastProjector, Volume, VideoWriter
from projection.plane_reader import read_plane_rows


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nrrd", help="3D volume in NRRD format")
    parser.add_argument("csv", help="planes, one per row: 3,a,b,c or 4,a1,b,a2,c "
                                    "(the lecturer's format)")
    parser.add_argument("mp4", help="output video, one frame per plane")
    parser.add_argument("--fps", type=float, default=10, help="video frame rate (default: 10)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.mp4.lower().endswith(".mp4"):
        print(f"Error: output file must end with .mp4, got {args.mp4}", file=sys.stderr)
        return 1
    try:
        volume = Volume.from_nrrd(args.nrrd)
        print(f"Volume: {volume.shape} {volume.dtype}")

        rows = read_plane_rows(args.csv)
        invalid = [row for row in rows if row.plane is None]
        for row in invalid:
            print(f"Warning: {args.csv}, line {row.line_number}: {row.error} "
                  "(this frame will be black)", file=sys.stderr)
        if len(invalid) == len(rows):
            raise ValueError(f"{args.csv}: no valid planes")
        print(f"Planes: {len(rows)} ({len(invalid)} invalid)")

        # one frame per CSV row, in order; an invalid row gets a black frame
        projector = FastProjector()
        black = np.zeros((1, 1), dtype=volume.dtype)
        start = time.perf_counter()
        frames = [projector.project(volume, row.plane) if row.plane is not None else black
                  for row in rows]
        elapsed = max(time.perf_counter() - start, 1e-9)
        projected = len(rows) - len(invalid)
        print(f"Projected {projected} frames in {elapsed:.3f}s "
              f"({projected / elapsed:.1f} fps)")

        VideoWriter(fps=args.fps).write(frames, args.mp4)
        print(f"Saved: {args.mp4}")
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except MemoryError:
        print("Error: not enough memory (too many or too large planes)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
