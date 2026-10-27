import sys
import time
import nrrd
import numpy as np
import imageio.v2 as imageio

EPS = 0.001


def validate_plane(a, a_second, b, c, Dab, Dac, Dbc):
    # בדיקות קלט תקין
    if Dab == 0 or Dac == 0:
        raise ValueError("Side lengths must be greater than zero")

    if abs(Dab - Dac) >= EPS:
        raise ValueError("Side lengths must differ by less than 0.001")

    # בדיקות נוספות לטרפז (4 נקודות)
    if not np.array_equal(a, a_second):
        if np.linalg.norm(a_second - a) > Dbc + EPS:
            raise ValueError("dist(a1,a2) must be <= dist(b,c)")

        # ארבע הנקודות על אותו מישור: הנפח שהן פורשות הוא 0
        normal = np.cross(b - a, c - a)
        if abs(np.dot(normal, a_second - a)) / np.linalg.norm(normal) >= EPS:
            raise ValueError("All 4 points must lie on the same plane")


def project_plane(M, a, b, c, a_second=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    a_second = a.copy() if a_second is None else np.asarray(a_second, dtype=float)

    Dab = np.linalg.norm(b - a)
    Dac = np.linalg.norm(c - a_second)
    Dbc = np.linalg.norm(c - b)
    validate_plane(a, a_second, b, c, Dab, Dac, Dbc)

    # וקטורי כיוון מנורמלים
    v_ab = (b - a) / Dab
    v_ac = (c - a_second) / Dac

    height = int(Dab)
    width = int(Dbc) + 1

    # כל ה-a1 וה-a2 של כל השורות בבת אחת (צעד לפני דגימה, כמו באלגוריתם)
    steps = np.arange(1, height + 1)[:, None]
    A1 = a + steps * v_ab
    A2 = a_second + steps * v_ac

    d12 = np.linalg.norm(A2 - A1, axis=1)
    dp = (A2 - A1) / np.where(d12 > 0, d12, 1)[:, None]

    # לכל פיקסל בפלט (y, x) מחשבים מאיזה x1 הוא הגיע: x1 = x - offset
    offset = ((Dbc - d12) / 2).astype(np.int32)
    x1 = np.arange(width, dtype=np.int32)[None, :] - offset[:, None]
    valid = (x1 >= 0) & (x1 <= d12[:, None].astype(np.int32)) & (d12[:, None] > 0)

    # מחשבים ציר אחרי ציר ובונים אינדקס שטוח לתוך M
    x1f = x1.astype(np.float32)
    flat = np.zeros((height, width), dtype=np.int32)   # sx*sy*sz < 2^30
    coord = np.empty((height, width), dtype=np.float32)
    for axis, size in enumerate(M.shape):
        np.multiply(x1f, np.float32(dp[:, axis])[:, None], out=coord)
        coord += np.float32(A1[:, axis])[:, None]
        np.rint(coord, out=coord)
        valid &= (coord >= 0) & (coord < size)      # p מחוץ ל-M -> v=0
        flat *= size
        flat += coord.astype(np.int32)

    # פיקסלים לא חוקיים מקבלים אינדקס 0 ואחר כך מאופסים
    flat *= valid
    E = M.ravel().take(flat)
    E *= valid
    return E


def read_planes(csv_path):
    planes = []
    with open(csv_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                numbers = [float(value) for value in line.split(",")]
            except ValueError:
                if line_number == 1:
                    continue        # שורת כותרת
                raise ValueError(f"Line {line_number}: non-numeric value")

            if len(numbers) == 9:
                a, b, c = numbers[0:3], numbers[3:6], numbers[6:9]
                planes.append((a, b, c, None))
            elif len(numbers) == 12:
                a, a_second = numbers[0:3], numbers[3:6]
                b, c = numbers[6:9], numbers[9:12]
                planes.append((a, b, c, a_second))
            else:
                raise ValueError(f"Line {line_number}: must contain 9 or 12 numbers")

    if not planes:
        raise ValueError("CSV file contains no planes")
    return planes


def pad_to_size(image, height, width):
    # ריפוד ממורכז כדי שהתמונה לא תקפוץ בין פריימים
    pad_h = height - image.shape[0]
    pad_w = width - image.shape[1]
    top, left = pad_h // 2, pad_w // 2
    return np.pad(image, ((top, pad_h - top), (left, pad_w - left)), mode="constant")


def frames_to_video(frames, mp4_path, fps=2):
    low = min(frame.min() for frame in frames)
    high = max(frame.max() for frame in frames)

    # כל הפריימים בוידאו חייבים להיות באותו גודל (כפולה של 16 עבור הקידוד)
    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    height += (-height) % 16
    width += (-width) % 16

    video_frames = []
    for frame in frames:
        if high > low:
            normalized = (frame.astype(np.float64) - low) / (high - low)
        else:
            normalized = np.zeros(frame.shape)
        image = (normalized * 255).astype(np.uint8)
        video_frames.append(pad_to_size(image, height, width))

    imageio.mimwrite(mp4_path, video_frames, fps=fps)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python Ex_1.py file.nrrd file.csv file.mp4")
        sys.exit(1)

    nrrd_path, csv_path, mp4_path = sys.argv[1], sys.argv[2], sys.argv[3]

    M, header = nrrd.read(nrrd_path)
    M = np.ascontiguousarray(M)
    print("Volume dimensions:", M.shape, M.dtype)

    planes = read_planes(csv_path)

    start = time.perf_counter()
    frames = [project_plane(M, a, b, c, a_second) for a, b, c, a_second in planes]
    elapsed = time.perf_counter() - start
    print(f"Projected {len(frames)} frames in {elapsed:.3f}s "
          f"({len(frames) / elapsed:.1f} fps)")

    frames_to_video(frames, mp4_path)
    print("Saved:", mp4_path)
