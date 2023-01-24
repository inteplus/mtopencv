"""A module dealing with image croppings and image crops.

Croppings and crops are understood as the followings. Cropping is the act of cutting off parts of
an image to form a smaller image, and maybe with a different resolution. Hence, a cropping is
analogous to an image transformation. A crop is the result of cropping an image. Hence, a crop is
like an image transform.
"""

from mt import tp, np, geo2d
from mt.base.deprecated import deprecated_func

from .warping import do_warp_image


__all__ = [
    "Cropping",
    "weight2crop",
    "estimate_cropping",
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
    cropres : list
        pair of `[width, height]` defining the resolution of the crop after being extracted from
        the source image
    crop : mt.geo2dRect, optional
        A different name for argument 'window'. For backward compatibility only.
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
        return "Cropping(imgres={}, window={}, cropres={})".format(
            self.imgres, self.window, self.cropres
        )

    def to_json(self):
        return {
            "imgres": self.imgres,
            "window": self.window.to_json(),
            "cropres": self.cropres,
        }

    @classmethod
    def from_json(cls, json_obj):
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

        Returns
        -------
        tfm : mt.geo2d.affine.Aff2d
            output 2D transformation
        """

        dst_rect = geo2d.Rect(0, 0, self.cropres[0], self.cropres[1])
        return geo2d.rect2rect(self.window, dst_rect)

    def get_img2crop_tfm_tf(self):
        """Returns the 2D affine transformation TF tensor mapping source pixels to crop pixels.

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

        Parameters
        ----------
        other : Cropping
            another cropping whose imgres is the same as the current cropres

        Returns
        -------
        Cropping
            the output composite image cropping, whose imgres is the same as that of self, and
            cropres is the same as that of other.
        """

        if self.cropres != other.imgres:
            raise ValueError(
                "The cropres of the current cropping {} is different from the imgres of the other cropping {}.".format(
                    self.cropres, other.imgres
                )
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
        """

        if self.imgres != other.imgres:
            raise ValueError(
                "The imgres of the current cropping {} is different from the imgres of the other cropping {}.".format(
                    self.imgres, other.imgres
                )
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
            input image from which the cropping takes place. It must have the same resolution as
            the imgres of the cropping.
        out_image : numpy.ndarray, optional
            output image to be cropped and resized to. If provided, it must have the same
            resolution as the cropres of the cropping. Otherwise, one is generated with the same
            dtype and number of channels as the input image, and with the same cropres of the
            cropping.
        inter_mode : {'nearest', 'bilinear'}
            interpolation mode. 'nearest' means nearest neighbour interpolation. 'bilinear' means
            bilinear interpolation
        border_mode : {'constant', 'replicate'}
            border filling mode. 'constant' means filling zero constant. 'replicate' means
            replicating last pixels in each dimension.

        Notes
        -----
        Since we use OpenCV for warping, the maximum number of channels is 4.
        """

        if in_image.shape[2] > 4:
            raise NotImplementedError(
                "OpenCV requires the maximum number of channels be 4. {} given.".format(
                    in_image.shape[2]
                )
            )

        if False:
            in_imgres = [in_image.shape[1], in_image.shape[0]]
            if in_imgres != self.imgres:
                raise ValueError(
                    "Expect the imgres to be {}. But {} given.".format(
                        self.imgres, in_imgres
                    )
                )

        if out_image is None:
            out_image = np.empty(
                (self.cropres[1], self.cropres[0], in_image.shape[2]),
                dtype=in_image.dtype,
            )

        out_imgres = [out_image.shape[1], out_image.shape[0]]
        if out_imgres != self.cropres:
            raise ValueError(
                "Expect the cropres to be {}. But {} given.".format(
                    self.cropres, out_imgres
                )
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
    alpha : float
        threshold to determine the level set beta such that the number of pixels whose value is
        greather than or equal to beta is greater than or equal to alpha*total weight.
    thresh : float
        threshold, below which the weight is set to zero
    square : bool
        whether or not to return a square or a rectangle
    padding : float
        percentage of padding compared on each dimension to make the returning rect larger than
        necessary (to make it convincing for food recognition for example)

    Returns
    -------
    mt.geo2d.rect.Rect
        a Rect such that all pixels whose values above beta (see above) are included, and that the
        total area including padding is as small as possible. If square is True, the returning
        rectangle is a square.
    """

    if alpha < 0 or alpha >= 1:
        raise ValueError("Alpha must be in interval [0,1). Got {}.".format(alpha))

    if padding < 0:
        raise ValueError("Padding must be non-negative. Got {}.".format(padding))

    if thresh < 0:
        raise ValueError("Threshold must be non-negative. Got {}.".format(thresh))

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
