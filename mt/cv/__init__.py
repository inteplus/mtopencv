"""Additional utitlities dealing with OpenCV for Python.

Instead of:

.. code-block:: python

   import cv2

You do:

.. code-block:: python

   from mt import cv

It will import the OpenCV package plus the additional stuff implemented in :module:`mt.opencv`.

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
