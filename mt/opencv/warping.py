"""Affine-warping and cropping an image.

All functions write their result in place into a pre-allocated output image, whose resolution
defines the resolution of the result. Images are row-major numpy arrays of shape `(height, width)`
or `(height, width, nchannels)` with at most 4 channels, as required by OpenCV.
"""


from mt import tp, np
import mt.geo2d as g2
from . import cv2 as cv


__all__ = ["do_warp_image", "warp_image", "crop_image"]


def do_warp_image(
    out_image: tp.NDArray[np.uint8],
    in_image: tp.NDArray[np.uint8],
    inv_tfm: g2.Aff2d,
    inter_mode: str = "nearest",
    border_mode: str = "constant",
):
    """Warps an input image into an output image using an inverse transformation.

    For every pixel location `p` of `out_image`, the pixel value is sampled from `in_image` at
    location `inv_tfm(p)`. The result is written into `out_image` in place.

    Parameters
    ----------
    out_image : numpy.ndarray
        pre-allocated output image of shape `(height, width[, nchannels])`. It receives the result
        and its size defines the output resolution
    in_image : numpy.ndarray
        input image from which pixels are sampled
    inv_tfm : mt.geo2d.Aff2d
        2D transformation mapping pixel locations in the output image to pixel locations in the
        input image
    inter_mode : {'nearest', 'bilinear'}, optional
        interpolation mode. 'nearest' means nearest neighbour. Any other value is treated as
        'bilinear'. Default is 'nearest'.
    border_mode : {'constant', 'replicate'}, optional
        how to fill pixels that fall outside the input image. 'constant' means filling with zeros.
        Any other value is treated as 'replicate', which repeats the last pixel in each dimension.
        Default is 'constant'.

    See Also
    --------
    warp_image : takes the forward transformation instead
    crop_image : takes a crop window instead

    Examples
    --------
    >>> import numpy as np
    >>> import mt.geo2d as g2
    >>> from mt.opencv.warping import do_warp_image
    >>> a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    >>> out = np.zeros((2, 2), dtype=np.uint8)
    >>> do_warp_image(out, a, g2.translate2d(1, 1))  # out(x, y) = a(x + 1, y + 1)
    >>> out
    array([[ 5,  6],
           [ 9, 10]], dtype=uint8)
    """
    borderMode = (
        cv.BORDER_CONSTANT if border_mode == "constant" else cv.BORDER_REPLICATE
    )  # we fill zeros here
    interMode = cv.INTER_NEAREST if inter_mode == "nearest" else cv.INTER_LINEAR
    cv.warpAffine(
        in_image,
        inv_tfm.matrix[:2, :],
        dst=out_image,
        dsize=(out_image.shape[1], out_image.shape[0]),
        flags=cv.WARP_INVERSE_MAP | interMode,
        borderMode=borderMode,
    )


def warp_image(
    out_image: tp.NDArray[np.uint8],
    in_image: tp.NDArray[np.uint8],
    warp_tfm: g2.Aff2d,
    inter_mode: str = "nearest",
    border_mode: str = "constant",
):
    """Warps an input image into an output image using a forward transformation.

    The transformation `warp_tfm` maps input pixel locations to the unit square `[0,1]^2`. It is
    scaled to the resolution of `out_image` and inverted, and then the input image is warped
    accordingly by :func:`do_warp_image`. The result is written into `out_image` in place.

    Parameters
    ----------
    out_image : numpy.ndarray
        pre-allocated output image of shape `(height, width[, nchannels])`. It receives the result
        and its size defines the output resolution
    in_image : numpy.ndarray
        input image from which pixels are sampled
    warp_tfm : mt.geo2d.Aff2d
        2D transformation mapping pixel locations in the input image to the `[0,1]^2` square
    inter_mode : {'nearest', 'bilinear'}, optional
        interpolation mode. 'nearest' means nearest neighbour. Any other value is treated as
        'bilinear'. Default is 'nearest'.
    border_mode : {'constant', 'replicate'}, optional
        how to fill pixels that fall outside the input image. 'constant' means filling with zeros.
        Any other value is treated as 'replicate', which repeats the last pixel in each dimension.
        Default is 'constant'.

    See Also
    --------
    do_warp_image : takes the inverse transformation in output pixel units
    crop_image : takes a crop window instead
    """
    inv_tfm = ~(g2.scale2d(out_image.shape[1], out_image.shape[0]) * warp_tfm)
    return do_warp_image(
        out_image, in_image, inv_tfm, inter_mode=inter_mode, border_mode=border_mode
    )


def crop_image(
    out_image: tp.NDArray[np.uint8],
    in_image: tp.NDArray[np.uint8],
    crop_rect: g2.Rect,
    inter_mode: str = "nearest",
    border_mode: str = "constant",
):
    """Cuts a crop window out of an input image and resizes it into an output image.

    The window is a rectangle in the pixel coordinates of `in_image`. It is warped to fill
    `out_image` completely, so the output resolution can differ from the window size. The result is
    written into `out_image` in place.

    Parameters
    ----------
    out_image : numpy.ndarray
        pre-allocated output image of shape `(height, width[, nchannels])`. It receives the crop
        and its size defines the crop resolution
    in_image : numpy.ndarray
        input image from which the crop takes place
    crop_rect : mt.geo2d.Rect
        crop window, in pixel coordinates of the input image. It may extend beyond the image, in
        which case the outside is filled according to `border_mode`.
    inter_mode : {'nearest', 'bilinear'}, optional
        interpolation mode. 'nearest' means nearest neighbour. Any other value is treated as
        'bilinear'. Default is 'nearest'.
    border_mode : {'constant', 'replicate'}, optional
        how to fill pixels that fall outside the input image. 'constant' means filling with zeros.
        Any other value is treated as 'replicate', which repeats the last pixel in each dimension.
        Default is 'constant'.

    See Also
    --------
    warp_image : more general warping
    mt.opencv.imgcrop.Cropping : a reusable description of a crop

    Examples
    --------
    >>> import numpy as np
    >>> import mt.geo2d as g2
    >>> from mt.opencv.warping import crop_image
    >>> a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    >>> out = np.zeros((2, 2), dtype=np.uint8)
    >>> crop_image(out, a, g2.Rect(1, 1, 3, 3))  # window with corners (1,1) and (3,3)
    >>> out
    array([[ 5,  6],
           [ 9, 10]], dtype=uint8)
    """
    crop_tfm = g2.crop_rect(crop_rect)
    return warp_image(
        out_image, in_image, crop_tfm, inter_mode=inter_mode, border_mode=border_mode
    )
