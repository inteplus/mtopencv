"""Regression tests for mt.opencv.imgres and mt.opencv.imgcrop."""

import unittest

import numpy as np

import mt.opencv
from mt.opencv import imgres
from mt.opencv.imgcrop import Cropping
import mt.geo2d as g2


class TestRepoSource(unittest.TestCase):
    def test_source_is_repo(self):
        self.assertTrue(mt.opencv.__file__.startswith("/home/minhtri/gitcentre/mtopencv/"))


class TestImgres(unittest.TestCase):
    def test_make_thumbnail_large_43(self):
        thumb = imgres.make_thumbnail(np.zeros((480, 640, 3), dtype=np.uint8), large=True)
        self.assertEqual(thumb.image.shape, (576, 768, 3))
        self.assertEqual(thumb.meta["src_imgres"], [640, 480])

    def test_make_thumbnail_large_169(self):
        thumb = imgres.make_thumbnail(np.zeros((720, 1280, 3), dtype=np.uint8), large=True)
        self.assertEqual(thumb.image.shape, (576, 1024, 3))

    def test_make_thumbnail_default(self):
        thumb = imgres.make_thumbnail(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(thumb.image.shape, (288, 384, 3))

    def test_fhd_is_1080p(self):
        self.assertEqual(imgres.name2imgres["fhd"], [1920, 1080])
        self.assertEqual(str(imgres.aspect_ratio(imgres.name2imgres["fhd"])), "16/9")


class TestCroppingApply(unittest.TestCase):
    def test_apply_2d(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = Cropping([4, 4], g2.Rect(0, 0, 4, 4), [2, 2]).apply(img, inter_mode="nearest")
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.tolist(), [[0, 2], [8, 10]])

    def test_apply_2d_out_image(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = np.zeros((2, 2), dtype=np.uint8)
        res = Cropping([4, 4], g2.Rect(0, 0, 4, 4), [2, 2]).apply(
            img, out_image=out, inter_mode="nearest"
        )
        self.assertIs(res, out)
        self.assertEqual(out.tolist(), [[0, 2], [8, 10]])

    def test_apply_3d_unchanged(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
        out = Cropping([4, 4], g2.Rect(0, 0, 4, 4), [2, 2]).apply(img, inter_mode="nearest")
        self.assertEqual(out.shape, (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
