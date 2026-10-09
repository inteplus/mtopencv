"""A module dealing with image croppings and image crops.

Croppings and crops are understood as the followings. Cropping is the act of cutting off parts of
an image to form a smaller image, and maybe with a different resolution. Hence, a cropping is
analogous to an image transformation. A crop is the result of cropping an image. Hence, a crop is
like an image transform.

Resolutions (`imgres` and `cropres`) are lists `[width, height]`, while numpy images have shape
`(height, width, nchannels)`.

Examples
--------
>>> import numpy as np
>>> import mt.geo2d as g2
>>> from mt.opencv.imgcrop import Cropping
>>> cropping = Cropping([640, 480], g2.Rect(100, 100, 300, 200), [100, 50])
>>> cropping.apply(np.zeros((480, 640, 3), dtype=np.uint8)).shape
(50, 100, 3)
"""

from mt import tp, np, geo2d
from mt.base.deprecated import deprecated_func

from . import cv2 as cv
from .warping import do_warp_image


__all__ = [
    "Cropping",
    "weight2crop",
    "estimate_cropping",
    "ultralytics_letterbox",
]


class Cropping:
    """An image cropping, the act of cutting a image to a crop window and resizing it.

    Parameters
    ----------
    imgres : list
        pair of `[width, height]` of the source image
    window : mt.geo2d.Rect, optional
        the rectangle on the source image defining where to cut/crop. If not given, it is set to be
        the rectangle capturing the whole image.
    cropres : list, optional
        pair of `[width, height]` defining the resolution of the crop after being extracted from
        the source image. Default is `[1, 1]`.
    crop : mt.geo2d.Rect, optional
        A different name for argument 'window'. For backward compatibility only. It is used only if
        `window` is not given.

    Attributes
    ----------
    imgres : list
        resolution of the source image
    window : mt.geo2d.Rect
        crop window on the source image
    cropres : list
        resolution of the crop

    Examples
    --------
    >>> import numpy as np
    >>> import mt.geo2d as g2
    >>> from mt.opencv.imgcrop import Cropping
    >>> cropping = Cropping([640, 480], g2.Rect(100, 100, 300, 200), [100, 50])
    >>> cropping
    Cropping(imgres=[640, 480], window=Rect(x=100.0, y=100.0, w=200.0, h=100.0), cropres=[100, 50])
    >>> cropping.get_img2crop_tfm()
    Aff2d(offset=vec2( -50, -50 ), linear=mat2x2(( 0.5, 0 ), ( 0, 0.5 )))
    >>> cropping.apply(np.zeros((480, 640, 3), dtype=np.uint8)).shape
    (50, 100, 3)
    """

    def __init__(
        self,
        imgres: list,
        window: tp.Optional[geo2d.Rect] = None,
        cropres: list = [1, 1],
        crop: tp.Optional[geo2d.Rect] = None,
    ):
        self.imgres = imgres
        if window is None:
            window = crop
        self.window = (
            geo2d.Rect(0, 0, imgres[0], imgres[1]) if window is None else window
        )
        self.cropres = cropres

    def __repr__(self):
        """Returns a string showing the imgres, window and cropres."""
        return f"Cropping(imgres={self.imgres}, window={self.window}, cropres={self.cropres})"

    def to_json(self):
        """Dumps the cropping to a JSON-like object.

        Returns
        -------
        dict
            a dictionary with keys 'imgres', 'window' (a list `[min_x, min_y, max_x, max_y]`) and
            'cropres'

        See Also
        --------
        from_json : the inverse operation

        Examples
        --------
        >>> import mt.geo2d as g2
        >>> from mt.opencv.imgcrop import Cropping
        >>> Cropping([640, 480], g2.Rect(100, 100, 300, 200), [100, 50]).to_json()
        {'imgres': [640, 480], 'window': [100.0, 100.0, 300.0, 200.0], 'cropres': [100, 50]}
        """
        return {
            "imgres": self.imgres,
            "window": self.window.to_json(),
            "cropres": self.cropres,
        }

    @classmethod
    def from_json(cls, json_obj):
        """Loads a cropping from a JSON-like object produced by :func:`Cropping.to_json`.

        Parameters
        ----------
        json_obj : dict
            the serialised cropping. The window is read from key 'crop' if present (backward
            compatibility), otherwise from key 'window'.

        Returns
        -------
        Cropping
            the loaded cropping

        See Also
        --------
        to_json : the inverse operation

        Examples
        --------
        >>> import mt.geo2d as g2
        >>> from mt.opencv.imgcrop import Cropping
        >>> c = Cropping([640, 480], g2.Rect(100, 100, 300, 200), [100, 50])
        >>> Cropping.from_json(c.to_json()).window
        Rect(x=100.0, y=100.0, w=200.0, h=100.0)
        """
        window = geo2d.Rect.from_json(
            json_obj["crop" if "crop" in json_obj else "window"]
        )
        return Cropping(
            json_obj["imgres"],
            window,
            json_obj["cropres"],
        )

    def get_img2crop_tfm(self) -> geo2d.Aff2d:
        """Returns the 2D affine transformation mapping source pixels to crop pixels.

        The transformation maps the crop window of the source image to the rectangle
        `[0, cropres[0]] x [0, cropres[1]]`.

        Returns
        -------
        tfm : mt.geo2d.Aff2d
            output 2D transformation
        """

        dst_rect = geo2d.Rect(0, 0, self.cropres[0], self.cropres[1])
        return geo2d.rect2rect(self.window, dst_rect)

    def get_img2crop_tfm_tf(self):
        """Returns the 2D affine transformation TF tensor mapping source pixels to crop pixels.

        It is the same transformation as :func:`get_img2crop_tfm` but as a tensor. TensorFlow is
        imported lazily.

        Returns
        -------
        tfm : tensorflow.Tensor
            output 3x3 matrix representing the 2D affine transformation. The 3x3 matrix can be used
            in :func:`tensorflow_graphics.image.transformer.perspective_transform`.
        """

        # f_u(x,y) = crop_width * (x - min_x) / (max_x - min_x)
        # f_v(x,y) = crop_height* (y - min_y) / (max_y - min_y)

        from mt import tf

        # scaling
        sx = self.cropres[0] / self.window.w
        sy = self.cropres[1] / self.window.h

        # translation
        tx = -self.window.min_x * sx
        ty = -self.window.min_y * sy

        return tf.convert_to_tensor([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])

    def join(self, other):
        """Joins with another image cropping to form a composite image cropping.

        Suppose the current cropping maps image A to crop B and `other` maps image B to crop C. The
        function returns the cropping that maps image A directly to crop C.

        Parameters
        ----------
        other : Cropping
            another cropping whose imgres is the same as the current cropres

        Returns
        -------
        Cropping
            the output composite image cropping, whose imgres is the same as that of self, and
            cropres is the same as that of other.

        Raises
        ------
        ValueError
            if the cropres of the current cropping is different from the imgres of `other`

        See Also
        --------
        rebase : changes the source image instead

        Examples
        --------
        >>> import mt.geo2d as g2
        >>> from mt.opencv.imgcrop import Cropping
        >>> c1 = Cropping([640, 480], g2.Rect(100, 100, 300, 200), [100, 50])
        >>> c2 = Cropping([100, 50], g2.Rect(0, 0, 50, 25), [10, 5])
        >>> c1.join(c2)
        Cropping(imgres=[640, 480], window=Rect(x=100.0, y=100.0, w=100.0, h=50.0), cropres=[10, 5])
        """

        if self.cropres != other.imgres:
            raise ValueError(
                f"The cropres of the current cropping {self.cropres} is different from the imgres "
                f"of the other cropping {other.imgres}."
            )

        tfm = self.get_img2crop_tfm()
        min_pt = tfm >> other.window.min_pt
        max_pt = tfm >> other.window.max_pt
        return Cropping(
            self.imgres,
            geo2d.Rect(min_pt[0], min_pt[1], max_pt[0], max_pt[1]),
            other.cropres,
        )

    def rebase(self, other):
        """Rebases the source image.

        Suppose the current cropping maps window X of image A to image C and the `other` cropping
        maps window Y of image A to image B. The function returns a cropping that maps window Z of
        image B to image C, where Z is the transform of window X from image A to image B.

        Parameters
        ----------
        other : Cropping
            another cropping whose imgres is the same as the current imgres

        Returns
        -------
        Cropping
            the output rebased cropping, whose imgres is the same as the cropres of the `other`
            cropping, and cropres is the same as that of the current cropping.

        Raises
        ------
        ValueError
            if the imgres of the current cropping is different from the imgres of `other`

        See Also
        --------
        join : composes two croppings in sequence

        Examples
        --------
        >>> import mt.geo2d as g2
        >>> from mt.opencv.imgcrop import Cropping
        >>> c = Cropping([640, 480], g2.Rect(100, 100, 300, 200), [100, 50])
        >>> other = Cropping([640, 480], g2.Rect(0, 0, 320, 240), [320, 240])
        >>> c.rebase(other).imgres
        [320, 240]
        """

        if self.imgres != other.imgres:
            raise ValueError(
                f"The imgres of the current cropping {self.imgres} is different from the imgres "
                f"of the other cropping {other.imgres}."
            )

        tfm = other.get_img2crop_tfm()

        min_pt = tfm << self.window.min_pt
        max_pt = tfm << self.window.max_pt
        return Cropping(
            other.cropres,
            geo2d.Rect(min_pt[0], min_pt[1], max_pt[0], max_pt[1]),
            self.cropres,
        )

    def apply(
        self,
        in_image: np.ndarray,
        out_image: tp.Optional[np.ndarray] = None,
        inter_mode: str = "bilinear",
        border_mode: str = "replicate",
    ) -> np.ndarray:
        """Applies the cropping to an image and returns the crop.

        Parameters
        ----------
        in_image : numpy.ndarray
            input image from which the cropping takes place, of shape `(height, width, nchannels)`
            with at most 4 channels, or of shape `(height, width)` for a single channel. It should
            have the same resolution as the imgres of the cropping (this is not checked).
        out_image : numpy.ndarray, optional
            output image to be cropped and resized to, of the same dimensionality as the input
            image. If provided, it must have the same resolution as the cropres of the cropping.
            Otherwise, one is generated with the same dtype and number of channels as the input
            image, and with the same cropres of the cropping.
        inter_mode : {'nearest', 'bilinear'}, optional
            interpolation mode. 'nearest' means nearest neighbour interpolation. 'bilinear' means
            bilinear interpolation. Default is 'bilinear'.
        border_mode : {'constant', 'replicate'}, optional
            border filling mode. 'constant' means filling zero constant. 'replicate' means
            replicating last pixels in each dimension. Default is 'replicate'.

        Returns
        -------
        numpy.ndarray
            the crop, which is `out_image` if provided, with shape `(cropres[1], cropres[0],
            nchannels)` (or `(cropres[1], cropres[0])` for a 2D input image)

        Raises
        ------
        NotImplementedError
            if the input image has more than 4 channels
        ValueError
            if the resolution of `out_image` is different from the cropres of the cropping

        Notes
        -----
        Since we use OpenCV for warping, the maximum number of channels is 4.

        The cropping object is also callable, with `cropping(in_image)` being the same as
        `cropping.apply(in_image)`.

        Examples
        --------
        >>> import numpy as np
        >>> import mt.geo2d as g2
        >>> from mt.opencv.imgcrop import Cropping
        >>> img = np.arange(16, dtype=np.uint8).reshape(4, 4, 1)
        >>> Cropping([4, 4], g2.Rect(0, 0, 4, 4), [2, 2]).apply(img, inter_mode="nearest")[:, :, 0]
        array([[ 0,  2],
               [ 8, 10]], dtype=uint8)
        """

        if in_image.ndim == 2:  # single-channel image of shape (height, width)
            out2d = self.apply(
                in_image[:, :, None],
                out_image=None if out_image is None else out_image[:, :, None],
                inter_mode=inter_mode,
                border_mode=border_mode,
            )
            return out2d[:, :, 0] if out_image is None else out_image

        if in_image.shape[2] > 4:
            raise NotImplementedError(
                f"OpenCV requires the maximum number of channels be 4. {in_image.shape[2]} given."
            )

        if False:
            in_imgres = [in_image.shape[1], in_image.shape[0]]
            if in_imgres != self.imgres:
                raise ValueError(
                    f"Expect the imgres to be {self.imgres}. But {in_imgres} given."
                )

        if out_image is None:
            out_image = np.empty(
                (self.cropres[1], self.cropres[0], in_image.shape[2]),
                dtype=in_image.dtype,
            )

        out_imgres = [out_image.shape[1], out_image.shape[0]]
        if out_imgres != self.cropres:
            raise ValueError(
                f"Expect the cropres to be {self.cropres}. But {out_imgres} given."
            )

        inv_tfm = ~self.get_img2crop_tfm()
        do_warp_image(
            out_image,
            in_image,
            inv_tfm,
            inter_mode=inter_mode,
            border_mode=border_mode,
        )

        return out_image

    __call__ = apply  # acronym


def weight2crop(
    weight_image: np.ndarray,
    alpha: float = 0.98,
    thresh: float = 0.0,
    square: bool = True,
    padding: float = 0.0,
) -> geo2d.Rect:
    """Estimates a crop that covers a minimum percentage of the total weight.

    Parameters
    ----------
    weight_image : numpy.ndarray
        a 2D weight image with shape (height, width) and every pixel has a non-negative weight
    alpha : float, optional
        threshold in `[0, 1)` to determine the level set beta such that the number of pixels whose
        value is greater than or equal to beta is greater than or equal to alpha*total weight.
        Default is 0.98.
    thresh : float, optional
        non-negative threshold, below which the weight is set to zero. Default is 0.
    square : bool, optional
        whether or not to return a square or a rectangle. Default is True.
    padding : float, optional
        non-negative percentage of padding compared on each dimension to make the returning rect
        larger than necessary (to make it convincing for food recognition for example).
        Default is 0.

    Returns
    -------
    mt.geo2d.Rect
        a Rect such that all pixels whose values above beta (see above) are included, and that the
        total area including padding is as small as possible. If square is True, the returning
        rectangle is a square. If the total weight is (almost) zero, a null rectangle `Rect(0, 0, 0,
        0)` is returned. The rectangle is in pixel coordinates `(x, y)` and may extend beyond the
        image when `square` or `padding` is used.

    Raises
    ------
    ValueError
        if `alpha` is not in `[0, 1)`, or `padding` or `thresh` is negative

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.imgcrop import weight2crop
    >>> w = np.zeros((10, 10), dtype=np.float32)
    >>> w[2:5, 3:8] = 1.0
    >>> weight2crop(w, square=False)
    Rect(x=3.0, y=2.0, w=5.0, h=3.0)
    >>> weight2crop(w)
    Rect(x=3.0, y=1.0, w=5.0, h=5.0)
    """

    if alpha < 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in interval [0,1). Got {alpha}.")

    if padding < 0:
        raise ValueError(f"Padding must be non-negative. Got {padding}.")

    if thresh < 0:
        raise ValueError(f"Threshold must be non-negative. Got {thresh}.")

    weight_image = np.where(weight_image >= thresh, weight_image, 0.0)

    # determine beta
    bin_cnt = 100
    hist, bins = np.histogram(weight_image, bins=bin_cnt, density=False)
    # print(hist)
    # print(bins)
    total_weight = np.dot(hist, bins[:bin_cnt])
    if abs(total_weight) < 1e-7:
        return geo2d.Rect(0, 0, 0, 0)  # null rect

    # print(total_weight)
    weight_thresh = alpha * total_weight
    # print(weight_thresh)
    sum_weight = 0
    beta = 0
    for i in range(bin_cnt - 1, -1, -1):
        sum_weight += hist[i] * bins[i]
        if sum_weight >= weight_thresh:
            beta = bins[i]
            break
    # print(beta)

    # determine the minimum rect
    ys, xs = np.where(weight_image >= beta)
    r0 = geo2d.Rect(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    # print(r0)
    if square:
        # adjust to make it a square with the same center
        cx = r0.cx
        cy = r0.cy
        r = max(r0.w, r0.h) / 2
        r0 = geo2d.Rect(cx - r, cy - r, cx + r, cy + r)
        # print(r0)

    # padding
    cx = r0.cx
    cy = r0.cy
    w2 = r0.w * (1 + padding) / 2
    h2 = r0.h * (1 + padding) / 2
    r0 = geo2d.Rect(cx - w2, cy - h2, cx + w2, cy + h2)
    # print(r0)
    return r0


def estimate_cropping(
    mask_cropping: Cropping,
    mask_crop: np.ndarray,
    out_cropres: list,
    alpha: float = 0.98,
    thresh: float = 0.0,
    square: bool = True,
    no_subpixel: bool = True,
    try_to_fit: bool = True,
) -> Cropping:
    """Estimates a cropping to contain almost all the content of a mask crop.

    The problem the function addresses is as follows. Suppose on a mask image space there is a
    cropping and its corresponding mask crop. Mask values are non-negative and anything outside the
    mask crop is treated to have 0 mask value. The goal is to estimate another cropping on the mask
    image space such that if we apply the new cropping, the total mask values in the new crop is to
    be not less than alpha times the total mask values on the image space.

    The solution involves building a level-set function, finding the optimal level, then bounding
    on any pixel not lower than that level.

    Parameters
    ----------
    mask_cropping : Cropping
        the original mask cropping
    mask_crop : numpy.ndarray
        a rank-2 array of shape `(H, W)` (matching the cropres of `mask_cropping`) representing the
        corresponding mask crop
    out_cropres: list
        pair `[crop_width, crop_height]` defining the cropres of the desired output cropping
    alpha : float
        threshold to determine the level set beta such that the sum of mask values of selected
        pixels is not less than alpha times the total mask value. A pixel is selected if its mask
        value is not less than beta.
    thresh : float
        threshold, below which the mask value is set to zero
    square : bool
        whether or not to the resultant crop window is square or rectangle
    no_subpixel : bool
        whether or not each of the output pixels must be at least as big as an input pixel
    try_to_fit : bool
        whether or not to try adjust the crop window to fit in the image resolution

    Returns
    -------
    out_cropping : Cropping
        the output cropping whose imgres matches the imgres of `mask_cropping` and whose cropres
        matches `out_cropres`. The crop window itself is either square or rectangle according to
        argument `square` and it is not guaranteed that the crop contains only pixels inside the
        image with imgres of `mask_cropping` as the resolution.
    """

    window = weight2crop(
        mask_crop, alpha=alpha, thresh=thresh, square=False, padding=0.0
    )

    src_imgres = mask_cropping.imgres

    cropping_on_mask = Cropping(
        mask_cropping.cropres, window=window, cropres=out_cropres
    )
    out_cropping = mask_cropping.join(cropping_on_mask)
    cx, cy = out_cropping.window.center_pt
    w = out_cropping.window.w
    h = out_cropping.window.h

    if no_subpixel:
        if w < out_cropres[0]:
            w = out_cropres[0]
        if h < out_cropres[1]:
            h = out_cropres[1]

    if square:
        h = w = max(w, h)

    hh = h * 0.5
    hw = w * 0.5

    if try_to_fit:
        if w >= src_imgres[0]:
            cx = src_imgres[0] * 0.5
        else:
            cx = float(np.clip(cx, hw, src_imgres[0] - hw))
        if h >= src_imgres[1]:
            cy = src_imgres[1] * 0.5
        else:
            cy = float(np.clip(cy, hh, src_imgres[1] - hh))

    window = geo2d.Rect(cx - hw, cy - hh, cx + hw, cy + hh)
    out_cropping = Cropping(src_imgres, window=window, cropres=out_cropres)

    return out_cropping


def ultralytics_letterbox(
    image: np.ndarray, new_imgres: list = [640, 640]
) -> tp.Tuple[np.ndarray, Cropping]:
    """Ultralytics style letterbox resizing.

    The image is scaled to fit in the new resolution while keeping the aspect ratio, and then
    padded on every side such that the image is always at the center and the padding is minimal.
    The padding is filled with the gray value 114 in every channel.

    Parameters
    ----------
    image : numpy.ndarray
        input image of shape `(height, width, nchannels)` to be letterbox resized
    new_imgres : list, optional
        pair of `[width, height]` defining the desired output image resolution. Default is
        `[640, 640]`.

    Returns
    -------
    numpy.ndarray
        the output letterbox resized image, of shape `(new_imgres[1], new_imgres[0], nchannels)`
    Cropping
        the cropping mapping from the letterboxed image used as input to YOLO models to the
        original image.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.imgcrop import ultralytics_letterbox
    >>> img = np.zeros((100, 200, 3), dtype=np.uint8)
    >>> out, cropping = ultralytics_letterbox(img, [64, 64])
    >>> out.shape, out[0, 0].tolist()
    ((64, 64, 3), [114, 114, 114])
    >>> cropping
    Cropping(imgres=[64, 64], window=Rect(x=0.0, y=16.0, w=64.0, h=32.0), cropres=[200, 100])
    """

    src_h, src_w = image.shape[0], image.shape[1]
    dst_w, dst_h = new_imgres[0], new_imgres[1]

    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)

    pad_w = dst_w - new_w
    pad_h = dst_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    letterbox_image = cv.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv.BORDER_CONSTANT,
        value=[114, 114, 114],
    )

    window = geo2d.Rect(pad_left, pad_top, pad_left + new_w, pad_top + new_h)
    cropping = Cropping(
        imgres=[dst_w, dst_h],
        window=window,
        cropres=[src_w, src_h],
    )

    return letterbox_image, cropping
