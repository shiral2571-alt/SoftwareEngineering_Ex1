import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from projection import VideoWriter, pad_to_size, read_planes

import Ex0


class TempDirTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write_file(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        return path


class TestReadPlanes(TempDirTestCase):

    def test_triangles_trapezoids_header_and_blank_lines(self):
        path = self.write_file("p.csv", "ax,ay,az,bx,by,bz,cx,cy,cz\n"
                                        "0,0,0,3,4,0,-3,4,0\n"
                                        "\n"
                                        "1,0,0,3,0,0,0,4,0,4,4,0\n")
        planes = read_planes(path)
        self.assertEqual([p.is_triangle for p in planes], [True, False])

    def test_invalid_plane_reports_line_number(self):
        path = self.write_file("p.csv", "0,0,0,3,4,0,-3,4,0\n0,0,0,1,0,0,0,5,0\n")
        with self.assertRaisesRegex(ValueError, "line 2"):
            read_planes(path)

    def test_wrong_number_of_values(self):
        path = self.write_file("p.csv", "1,2,3,4\n")
        with self.assertRaisesRegex(ValueError, "line 1"):
            read_planes(path)

    def test_empty_file(self):
        with self.assertRaises(ValueError):
            read_planes(self.write_file("p.csv", "\n\n"))


class TestVideoWriter(unittest.TestCase):

    def test_frames_of_different_sizes_get_common_padded_size(self):
        frames = [np.ones((5, 7), np.int16), np.ones((20, 3), np.int16)]
        images = VideoWriter.prepare(frames)
        self.assertEqual({img.shape for img in images}, {(32, 16)})
        self.assertTrue(all(img.dtype == np.uint8 for img in images))

    def test_global_normalization(self):
        images = VideoWriter.prepare([np.full((2, 2), 100), np.full((2, 2), 300)])
        self.assertEqual(images[0].max(), 0)
        self.assertEqual(images[1].max(), 255)

    def test_constant_frames_do_not_divide_by_zero(self):
        images = VideoWriter.prepare([np.full((3, 3), 7)])
        self.assertFalse(images[0].any())

    def test_pad_is_centered(self):
        padded = pad_to_size(np.ones((2, 2), np.uint8), 4, 6)
        np.testing.assert_array_equal(np.argwhere(padded)[0], [1, 2])


class TestDemoEndToEnd(TempDirTestCase):
    """Runs Ex0.main exactly like `python Ex0.py vol.nrrd planes.csv out.mp4`."""

    def make_volume(self):
        import nrrd
        path = os.path.join(self.dir, "vol.nrrd")
        x, y, z = np.indices((32, 32, 16))
        nrrd.write(path, ((x + y + z) % 256).astype(np.uint8))
        return path

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = Ex0.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_video_has_one_frame_per_csv_row(self):
        import imageio.v2 as imageio
        rows = [f"0,0,{z},31,0,{z},0,31,{z},31,31,{z}" for z in range(0, 16, 2)]
        rows.append("16,0,8,0,20,8,32,20,8")
        csv_path = self.write_file("planes.csv", "\n".join(rows) + "\n")
        mp4_path = os.path.join(self.dir, "out.mp4")

        code, out, _ = self.run_main(self.make_volume(), csv_path, mp4_path)

        self.assertEqual(code, 0)
        self.assertIn("fps", out)
        with imageio.get_reader(mp4_path) as reader:
            self.assertEqual(reader.count_frames(), len(rows))

    def test_missing_input_file_returns_error_code(self):
        csv_path = self.write_file("planes.csv", "0,0,0,3,4,0,-3,4,0\n")
        code, _, err = self.run_main(os.path.join(self.dir, "missing.nrrd"), csv_path, "o.mp4")
        self.assertEqual(code, 1)
        self.assertIn("Error", err)

    def test_invalid_plane_returns_error_code(self):
        csv_path = self.write_file("planes.csv", "0,0,0,1,0,0,0,5,0\n")
        code, _, err = self.run_main(self.make_volume(), csv_path, "o.mp4")
        self.assertEqual(code, 1)
        self.assertIn("line 1", err)


if __name__ == "__main__":
    unittest.main()
