'''OpenCV package with some extra functionalities implemented by Minh-Tri Pham.

Examples
--------
>>> from mt import cv

'''

from mt.base import logger

from .version import version as __version__

try:
    import cv2
except ImportError:
    logger.error("IMPORT: OpenCV for Python is not detected. Please install a version of OpenCV for Python.")
    raise
