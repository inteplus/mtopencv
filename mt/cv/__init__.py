"""Additional utilities dealing with OpenCV for Python.

Instead of:

.. code-block:: python

   import cv2

You do:

.. code-block:: python

   from mt import cv

It will import the OpenCV package, re-exporting every public name of :mod:`cv2` (so that
`cv.imread`, `cv.resize`, etc. work as usual), plus the additional stuff implemented in
:mod:`mt.opencv`, such as :class:`mt.opencv.image.Image`, :class:`mt.opencv.imgcrop.Cropping`,
:func:`mt.opencv.warping.crop_image`, the polygon functions of :mod:`mt.opencv.polygon` and
the image resolution helpers of :mod:`mt.opencv.imgres`.

Please see `opencv`_ package for Python for more details.

.. _opencv:
   https://docs.opencv.org/
"""

import cv2

for key in cv2.__dict__:
    if not key.startswith("__") and not key == "cv2":
        globals()[key] = getattr(cv2, key)
from cv2 import __version__
from mt.opencv import cv2, logger
from mt.opencv.polygon import *
from mt.opencv.warping import *
from mt.opencv.image import *
from mt.opencv.imgcrop import *
from mt.opencv.ansi import *
from mt.opencv import imgres
