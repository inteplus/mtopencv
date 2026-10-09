'''OpenCV package with some extra functionalities implemented by Minh-Tri Pham.

The package exposes the `cv2` module, and the submodules :mod:`~mt.opencv.image`,
:mod:`~mt.opencv.imgcrop`, :mod:`~mt.opencv.imgres`, :mod:`~mt.opencv.polygon`,
:mod:`~mt.opencv.warping` and :mod:`~mt.opencv.ansi`. The easiest way to use all of them at once
is via the :mod:`mt.cv` namespace.

Examples
--------
>>> from mt import cv
'''

from mt.base import logger

from .version import version as __version__

try:
    import cv2
except ImportError:
    logger.error(
        "IMPORT: OpenCV for Python is not detected. "
        "Please install a version of OpenCV for Python."
    )
    raise
