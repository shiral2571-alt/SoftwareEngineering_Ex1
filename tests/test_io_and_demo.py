import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

from projection import VideoWriter, pad_to_size, read_planes
from projection.plane_reader import read_plane_rows

import Ex0

LECTURER_EXAMPLE = ("Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\r\n"
                    "3,200,200,0,0,200,400,400,200,400,,,\r\n"
                    "4,200,0,0,200,0,500,200,400,0,200,400,500\r\n"
                    "3,200,200,-30,-50,200,400,400,250,400,,,\r\n"
                    "4,200,50,0,200,0,500,200,350,0,200,400,500")


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

    def test_semicolons_tabs_and_spaces(self):
        path = self.write_file("p.csv", "0;0;0;3;4;0;-3;4;0\n"
                                        "0\t0\t0\t3\t4\t0\t-3\t4\t0\n"
                                        " 0 , 0 ,0, 3,4,0 ,-3,4,0 \r\n")
        self.assertEqual(len(read_planes(path)), 3)

    def test_empty_value_inside_a_row_is_an_error(self):
        for text, line in [("0,0,0,3,4,0,-3,4,0\n0,0,0,3,,0,-3,4,0\n", "line 2"),
                           ("0,0,,3,4,0,-3,4,0\n", "line 1")]:   # not mistaken for a header
            path = self.write_file("p.csv", text)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, line):
                read_planes(path)

    def test_lecturer_format(self):
        # the example file linked in the assignment: a count column, (a1, b, a2, c)
        # order for trapezoids, empty x4,y4,z4 for triangles; row 4 is not isosceles
        path = os.path.join(self.dir, "lecturer.csv")
        with open(path, "wb") as file:          # keep the CRLF line ends exactly
            file.write(LECTURER_EXAMPLE.encode())
        rows = read_plane_rows(path)
        self.assertEqual([r.line_number for r in rows], [2, 3, 4, 5])
        self.assertEqual([r.plane is not None for r in rows], [True, True, False, True])
        self.assertIn("Side lengths", rows[2].error)
        trapezoid = rows[3].plane
        np.testing.assert_array_equal(trapezoid.a1, [200, 50, 0])
        np.testing.assert_array_equal(trapezoid.b, [200, 0, 500])
        np.testing.assert_array_equal(trapezoid.a2, [200, 350, 0])
        np.testing.assert_array_equal(trapezoid.c, [200, 400, 500])

    def test_lecturer_format_count_must_match(self):
        path = self.write_file("p.csv", "4,0,0,0,3,4,0,-3,4,0,,,\n")
        with self.assertRaisesRegex(ValueError, "line 1"):
            read_planes(path)


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

    def test_percentile_clipping_saturates_outliers(self):
        frame = np.zeros((16, 16))    # already a multiple of 16: no padding
        frame[1:, :] = 100
        frame[0, 0] = 10_000          # one hot voxel
        image = VideoWriter.prepare([frame], clip_percentile=90)[0]
        self.assertEqual(image[0, 0], 255)
        self.assertEqual(image[5, 5], 255)   # 100 is the 90th percentile, not dark gray
        self.assertEqual(image[0, 1], 0)

    def test_constant_frames_do_not_divide_by_zero(self):
        images = VideoWriter.prepare([np.full((3, 3), 7)])
        self.assertFalse(images[0].any())

    def test_nan_voxels_are_black_and_do_not_spoil_the_window(self):
        frame = np.full((16, 16), 50.0)
        frame[0, 0] = np.nan
        frame[0, 1] = 100.0
        image = VideoWriter.prepare([frame])[0]
        self.assertEqual(image[0, 0], 0)
        self.assertEqual(image[0, 1], 255)

    def test_crop_black_margins_keeps_frames_aligned(self):
        from projection.video import crop_black_margins
        a = np.zeros((64, 64), np.uint8)
        b = np.zeros((64, 64), np.uint8)
        a[10, 20] = 255                   # content in different places of the two frames
        b[30, 40] = 255
        black = np.zeros((64, 64), np.uint8)
        cropped = crop_black_margins([a, black, b])
        self.assertEqual({img.shape for img in cropped}, {(32, 32)})   # 21x21 -> padded to 32
        (ya, xa), = np.argwhere(cropped[0])
        (yb, xb), = np.argwhere(cropped[2])
        self.assertEqual((yb - ya, xb - xa), (20, 20))                 # same offset as before
        self.assertFalse(cropped[1].any())
        self.assertEqual(crop_black_margins([black])[0].shape, (64, 64))   # all black: unchanged

    def test_invalid_fps(self):
        for fps in (0, -5, 0.01, float("nan"), float("inf"), 10_000):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                VideoWriter(fps=fps)

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

    def assert_clean_error(self, *args):
        code, _, err = self.run_main(*args)
        self.assertEqual(code, 1)
        self.assertIn("Error:", err)
        self.assertNotIn("Traceback", err)

    def test_lecturer_example_file(self):
        import imageio.v2 as imageio
        csv_path = os.path.join(self.dir, "lecturer.csv")
        with open(csv_path, "wb") as file:
            file.write(LECTURER_EXAMPLE.encode())
        mp4_path = os.path.join(self.dir, "out.mp4")

        code, out, err = self.run_main(self.make_volume(), csv_path, mp4_path)

        self.assertEqual(code, 0)
        self.assertIn("line 4", err)            # the non-isosceles triangle is reported
        self.assertIn("4 (1 invalid)", out)
        with imageio.get_reader(mp4_path) as reader:
            self.assertEqual(reader.count_frames(), 4)   # still one frame per row

    def test_corrupt_nrrd_file(self):
        csv_path = self.write_file("planes.csv", "0,0,0,3,4,0,-3,4,0\n")
        bad_nrrd = self.write_file("bad.nrrd", "this is not an nrrd file")
        self.assert_clean_error(bad_nrrd, csv_path, os.path.join(self.dir, "o.mp4"))

    def test_tiny_and_huge_planes(self):
        for row in ["0,0,0,0.5,0.3,0,0.5,-0.3,0", "0,0,0,0,1e7,1e7,0,1e7,-1e7"]:
            with self.subTest(row=row):
                csv_path = self.write_file("planes.csv", row + "\n")
                self.assert_clean_error(self.make_volume(), csv_path,
                                        os.path.join(self.dir, "o.mp4"))

    def test_bad_output_and_fps(self):
        csv_path = self.write_file("planes.csv", "0,0,0,3,4,0,-3,4,0\n")
        volume = self.make_volume()
        self.assert_clean_error(volume, csv_path, os.path.join(self.dir, "o.txt"))
        self.assert_clean_error(volume, csv_path, os.path.join(self.dir, "no", "o.mp4"))
        self.assert_clean_error(volume, csv_path, os.path.join(self.dir, "o.mp4"), "--fps", "0")


if __name__ == "__main__":
    unittest.main()
