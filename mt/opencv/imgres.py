"""A module to represent an image resolution to avoid misuse of cv2.Size and numpy.ndarray's shape.

An image resolution (imgres) is defined as a list [width, height], so that it can be smoothly
serialized using yaml. There are some functions in this module to convert between imgres and common
image resolutions types like equivalents of OpenCV's size format or a part of a row-major
numpy.ndarray's shape. Note that a row-major image array has shape `(height, width, ...)`, the
reverse order of an imgres.

A few common resolutions are used here, available by name in dictionary `name2imgres`. The
reference for resolutions is: https://en.wikipedia.org/wiki/List_of_common_resolutions

For CIF-based resolutions, see https://en.wikipedia.org/wiki/Low-definition_television

Examples
--------
>>> from mt.opencv import imgres
>>> imgres.name2imgres["qvga"]
[320, 240]
>>> imgres.shape2imgres((480, 640, 3))
[640, 480]
>>> imgres.imgres2shape([640, 480])
(480, 640)
"""

import fractions
import cv2

from mt import np, geo2d
from mt.base.deprecated import deprecated_func

from .image import Image

__all__ = [
    "name2imgres",
    "equal",
    "imgres2size",
    "imgres2shape",
    "size2imgres",
    "shape2imgres",
    "aspect_ratio",
    "get_center_window",
    "get_center_window_tfm",
    "get_center_window_tfm_tf",
    "make_thumbnail",
    "get_thumbnail_imgres",
]


# Reference: https://en.wikipedia.org/wiki/List_of_common_resolutions
name2imgres = {
    "qqvga": [160, 120],
    "sony_smartwatch": [128, 128],
    "hqvga": [240, 160],
    "nintendo_ds": [256, 192],
    "cga": [320, 200],
    "qvga": [320, 240],
    "cif": [384, 288],
    "ws_cif": [512, 288],
    "pal43": [768, 576],
    "pal169": [1024, 576],
    "uxga": [1600, 1200],
    "fhd": [1920, 1080],
}


def equal(imgres1, imgres2):
    """Checks if two imgresses are equal or not.

    Parameters
    ----------
    imgres1, imgres2 : list
        pairs of `[width, height]`

    Returns
    -------
    bool
        whether the widths and the heights are respectively equal

    Examples
    --------
    >>> from mt.opencv import imgres
    >>> imgres.equal([640, 480], (640, 480))
    True
    """
    return (imgres1[0] == imgres2[0]) and (imgres1[1] == imgres2[1])


def imgres2size(imgres):
    """Converts an imgres `[width, height]` into an equivalent OpenCV's size `(width, height)`.

    Returns
    -------
    tuple
        the pair `(width, height)`
    """
    return (imgres[0], imgres[1])


def imgres2shape(imgres):
    """Converts an imgres into an int tuple that can be used to define a row-major numpy.ndarray.

    Parameters
    ----------
    imgres : list
        pair of `[width, height]`

    Returns
    -------
    tuple
        the pair `(height, width)`

    See Also
    --------
    shape2imgres : the inverse operation
    """
    return (imgres[1], imgres[0])


def size2imgres(size):
    """Converts an equivalent of OpenCV's size `(width, height)` into imgres `[width, height]`.

    Returns
    -------
    list
        the pair `[width, height]`
    """
    return [size[0], size[1]]


def shape2imgres(shape):
    """Converts a part of a row-major numpy.ndarray's shape into imgres.

    Parameters
    ----------
    shape : tuple
        the shape of an image array, of the form `(height, width)` or `(height, width, ...)`. Only
        the first two items are used.

    Returns
    -------
    list
        the pair `[width, height]`

    See Also
    --------
    imgres2shape : the inverse operation
    """
    return [shape[1], shape[0]]


def aspect_ratio(imgres):
    """Gets the aspect ratio of the resolution.

    Parameters
    ----------
    imgres : list
        pair of `[width, height]` of integers

    Returns
    -------
    fractions.Fraction
        the exact width-over-height ratio

    Examples
    --------
    >>> from mt.opencv import imgres
    >>> imgres.aspect_ratio([1920, 1080])
    Fraction(16, 9)
    """
    return fractions.Fraction(imgres[0], imgres[1])


def get_center_window(aspect_ratio, src_imgres, alpha=1.0):
    """Returns a center window of the source image that preserves the width-over-height aspect
    ratio.

    The center window is defined in the following:

       - Its center is the center of the source image.
       - Its width is not more than the source width times alpha.
       - Its height is not more than the source height times alpha.
       - Its width-over-height aspect ratio is the same as the provided aspect ratio.
       - It is as large as possible.

    Parameters
    ----------
    aspect_ratio : float
        the input width-over-height aspect ratio
    src_imgres : list
        pair of `[width, height]` of the source image resolution
    alpha : float, optional
        a positive scalar telling how large the window can be. Default is 1.0.

    Returns
    -------
    rect : mt.geo2d.Rect
        the output center window, in pixel coordinates of the source image

    Raises
    ------
    ValueError
        if `alpha` is not positive

    Examples
    --------
    >>> from mt.opencv import imgres
    >>> imgres.get_center_window(1.0, [640, 480])
    Rect(x=80.0, y=0.0, w=480.0, h=480.0)
    >>> imgres.get_center_window(1.0, [640, 480], alpha=0.5)
    Rect(x=200.0, y=120.0, w=240.0, h=240.0)
    """

    if alpha <= 0:
        raise ValueError(f"A non-positive 'alpha' has been detected: {alpha}.")

    sw = src_imgres[0] * alpha
    sh = src_imgres[1] * alpha

    # make sure aspect ratio is preserved by trimming the longer dimension
    if sw > sh * aspect_ratio:  # width larger than height
        sw = sh * aspect_ratio
    else:
        sh = sw / aspect_ratio

    return geo2d.Rect(
        (src_imgres[0] - sw) / 2,
        (src_imgres[1] - sh) / 2,
        (src_imgres[0] + sw) / 2,
        (src_imgres[1] + sh) / 2,
    )


@deprecated_func(
    "1.9",
    suggested_func="mt.cv.ImgCrop",
    removed_version="2.0",
    docstring_prefix="    ",
)
def get_center_window_tfm(dst_imgres, src_imgres, alpha=1.0):
    """Returns a 2D affine transformation that maps pixels in a source image to pixels in a
    destination image that reflects a center window of the source image.

    The center window is defined in the following:

       - Its center is the center of the source image.
       - Its width is not more than the source width times alpha.
       - Its height is not more than the source height times alpha.
       - Its aspect ratio is the same as that of the destination image resolution.
       - It is as large as possible.

    The function returns a transformation. One can use a warping function to warp the image and then
    crop using the destination imgres.

    Parameters
    ----------
    dst_imgres : list
        pair of `[width, height]` of the destination image resolution
    src_imgres : list
        pair of `[width, height]` of the source image resolution
    alpha : float, optional
        a positive scalar telling how large the window can be. Default is 1.0.

    Returns
    -------
    tfm : mt.geo2d.Aff2d
        output 2D transformation

    Raises
    ------
    ValueError
        if `alpha` is not positive

    See Also
    --------
    get_center_window : returns the window itself
    mt.opencv.imgcrop.Cropping : the recommended replacement
    """

    src_rect = get_center_window(dst_imgres[0] / dst_imgres[1], src_imgres, alpha=alpha)
    dst_rect = geo2d.Rect(0, 0, dst_imgres[0], dst_imgres[1])
    from mt.geo2d import rect2rect

    return rect2rect(src_rect, dst_rect)


@deprecated_func(
    "1.9",
    suggested_func="mt.cv.ImgCrop",
    removed_version="2.0",
    docstring_prefix="    ",
)
def get_center_window_tfm_tf(dst_shape, src_shape, alpha=1.0):
    """Tensorflow version of :func:`get_center_window_tfm`.

    Unlike the original function, the function inputs and outputs tensors. Instead of imgreses,
    the function inputs shapes, which are in `[height, width]` order.

    Parameters
    ----------
    dst_shape : tensorflow.Tensor or list
        pair of `[height, width]` of the destination image resolution
    src_shape : tensorflow.Tensor or list
        pair of `[height, width]` of the source image resolution
    alpha : tensorflow.Tensor or float, optional
        a positive scalar telling how large the window can be. Default is 1.0.

    Returns
    -------
    tfm : tensorflow.Tensor
        output 3x3 matrix representing the 2D affine transformation mapping from source center
        window to destination image. The 3x3 matrix can be used in
        :func:`tensorflow_graphics.image.transformer.perspective_transform`.
    """

    from mt import tf

    aspect_ratio = dst_shape[1] / dst_shape[0]

    src1_shape = (alpha * src_shape[0]) * tf.convert_to_tensor([1.0, aspect_ratio])
    src2_shape = (alpha * src_shape[1]) / tf.convert_to_tensor([aspect_ratio, 1.0])
    src3_shape = tf.where(
        src_shape[1] > src_shape[0] * aspect_ratio, src1_shape, src2_shape
    )

    # src window has shape src3_shape and centered at origin of src image
    # dst window is just dst image

    # scaling
    sx = dst_shape[1] / src3_shape[1]
    sy = dst_shape[0] / src3_shape[0]

    # translation
    tx = (
        dst_shape[1] - src_shape[1] * sx
    ) / 2  # (src_shape[1]/2)*sx + tx = (dst_shape[1]/2)
    ty = (
        dst_shape[0] - src_shape[0] * sy
    ) / 2  # (src_shape[0]/2)*sy + ty = (dst_shape[0]/2)

    return tf.convert_to_tensor([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])


def get_thumbnail_imgres(raw_imgres: list, large: bool = False) -> list:
    """Gets the thumbnail resolution from the raw imgres.

    Only raw resolutions of aspect ratio 4:3 or 16:9 are supported. For 4:3, the thumbnail is 'cif'
    (or 'pal43' if `large`). For 16:9, it is 'ws_cif' (or 'pal169' if `large`). See `name2imgres`.

    Parameters
    ----------
    raw_imgres : list
        pair `[width, height]` representing the raw image resolution
    large : bool, optional
        whether or not to make a large thumbnail. Default is False.

    Returns
    -------
    thumb_imgres : list
        pair `[width, height]` representing the resolution of the thumbnail. It is the list stored
        in `name2imgres` (not a copy).

    Raises
    ------
    NotImplementedError
        if the aspect ratio of `raw_imgres` is neither 4:3 nor 16:9

    Examples
    --------
    >>> from mt.opencv import imgres
    >>> imgres.get_thumbnail_imgres([640, 480])
    [384, 288]
    >>> imgres.get_thumbnail_imgres([1280, 720], large=True)
    [1024, 576]
    """

    ar = aspect_ratio(raw_imgres)
    if ar == fractions.Fraction(4, 3):
        name = "pal43" if large else "cif"
    elif ar == fractions.Fraction(16, 9):
        name = "pal169" if large else "ws_cif"
    else:
        raise NotImplementedError(
            f"Aspect ratio {ar} not yet implemented for imgres {raw_imgres}."
        )

    return name2imgres[name]


def make_thumbnail(
    image: np.ndarray,
    large: bool = False,
    pixel_format: str = "rgb",
    extra_meta: dict = {},
) -> Image:
    """Makes a thumbnail out of an image.

    Only images of aspect ratio 4:3 or 16:9 are accepted. The thumbnail of a 4:3 image will be of
    resolution 'cif' (384x288) and that of a 16:9 image will be of resolution 'ws_cif' (512x288).
    If `large` is True, they are 'pal43' (768x576) and 'pal169' (1024x576) respectively. See
    attribute `name2imgres` of the module for more details.

    Parameters
    ----------
    image : numpy.ndarray
        an image of shape `(H, W, D)` where `1 <= D <= 4`
    large : bool, optional
        whether or not to make a large thumbnail ('pal43' or 'pal169'). Default is False.
    pixel_format : str, optional
        pixel format. To be passed as-is to :class:`mt.opencv.image.Image`. Default is 'rgb'.
    extra_meta : dict, optional
        extra metadata for the image. To be passed as-is to :class:`mt.opencv.image.Image`

    Returns
    -------
    mt.opencv.image.Image
        another image of shape `(288, 384, D)` or `(288, 512, D)` (or `(576, 768, D)` or
        `(576, 1024, D)` if `large`) with metadata. The metadata of the
        image contains key 'src_imgres' telling the resolution of the original image, plus any
        metadata provided by the `extra_meta` dictionary.

    Raises
    ------
    NotImplementedError
        if the aspect ratio of the image is neither 4:3 nor 16:9

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.imgres import make_thumbnail
    >>> make_thumbnail(np.zeros((480, 640, 3), dtype=np.uint8))
    cv.Image(image.shape=(288, 384, 3), pixel_format='rgb', meta={"src_imgres": [640, 480]})
    """

    imgres = shape2imgres(image.shape)
    thumb_imgres = get_thumbnail_imgres(imgres, large=large)
    img = cv2.resize(image, thumb_imgres)
    meta = extra_meta.copy()
    meta["src_imgres"] = imgres
    thumb = Image(img, pixel_format=pixel_format, meta=meta)

    return thumb
