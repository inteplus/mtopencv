"""A self-contained image: an array of pixels bundled with its pixel format and metadata.

The main class is :class:`Image`, which can be serialised to JSON or HDF5, and saved to or loaded
from a file with :func:`immsave` and :func:`immload`. Functions :func:`imload` and :func:`imsave`
are asynchronous wrappers around :func:`cv2.imread` and :func:`cv2.imwrite`, and
:func:`im_float2ubyte` and :func:`im_ubyte2float` convert between float and uint8 pixel values.

Images are numpy arrays of shape `(height, width, nchannels)`, or `(height, width)` for the
'gray' pixel format, with dtype uint8. The channel order is given by the pixel format.

Examples
--------
>>> import numpy as np
>>> from mt.opencv.image import Image
>>> img = Image(np.zeros((2, 3, 3), dtype=np.uint8), pixel_format="rgb", meta={"id": 1})
>>> img
cv.Image(image.shape=(2, 3, 3), pixel_format='rgb', meta={"id": 1})
"""

import cv2
import base64
import json

from mt import tp, np, path, aio, base

__all__ = [
    "PixelFormat",
    "Image",
    "immload_asyn",
    "immload",
    "immload_header_asyn",
    "immload_header",
    "immsave_asyn",
    "immsave",
    "imload",
    "imsave",
    "im_float2ubyte",
    "im_ubyte2float",
]


PixelFormat = {
    "rgb": 3,
    "bgr": 3,
    "rgba": 4,
    "bgra": 4,
    "argb": 4,
    "abgr": 4,
    "gray": 1,
}

_RGB_INDICES = {
    "rgb": (0, 1, 2),
    "bgr": (2, 1, 0),
    "rgba": (0, 1, 2),
    "bgra": (2, 1, 0),
    "argb": (1, 2, 3),
    "abgr": (3, 2, 1),
}


def _alpha_channel_index(pixel_format: str) -> int:
    return pixel_format.find("a")


def _to_cv_image(image, pixel_format: str):
    if pixel_format == "gray":
        return image
    r_idx, g_idx, b_idx = _RGB_INDICES[pixel_format]
    return np.ascontiguousarray(image[:, :, [b_idx, g_idx, r_idx]])


def _from_cv_image(image, pixel_format: str):
    if pixel_format == "gray":
        return image

    nchannels = PixelFormat[pixel_format]
    out = np.empty((image.shape[0], image.shape[1], nchannels), dtype=image.dtype)
    r_idx, g_idx, b_idx = _RGB_INDICES[pixel_format]
    out[:, :, r_idx] = image[:, :, 2]
    out[:, :, g_idx] = image[:, :, 1]
    out[:, :, b_idx] = image[:, :, 0]

    a_idx = _alpha_channel_index(pixel_format)
    if a_idx >= 0:
        out[:, :, a_idx] = 255

    return out


def _encode_jpeg(image, quality: tp.Optional[int]):
    if quality is None:
        retval, arr = cv2.imencode(".jpg", image)
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        retval, arr = cv2.imencode(".jpg", image, params)

    if not retval:
        raise RuntimeError(
            f"Unable to use OpenCV to jpg-encode the image of shape {image.shape}."
        )

    return arr.tobytes()


def _decode_jpeg(buf: bytes, gray: bool = False):
    flags = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    arr = np.frombuffer(buf, dtype=np.uint8)
    image = cv2.imdecode(arr, flags)
    if image is None:
        raise RuntimeError("Unable to use OpenCV to jpg-decode image bytes.")
    return image


class Image(object):
    """A self-contained image, where the meta-data associated with the image are kept together with
    the image itself.

    Parameters
    ----------
    image : numpy.ndarray
        a 2D image of shape `(height, width, nchannels)` or `(height, width)` with dtype uint8. It
        is
        converted into a C-contiguous array. Its channel order must follow `pixel_format`.
    pixel_format : str, optional
        one of the keys in the `PixelFormat` mapping, namely 'rgb', 'bgr', 'rgba', 'bgra', 'argb',
        'abgr' and 'gray'. The mapping gives the number of channels of each format. Default is
        'rgb'.
    meta : dict, optional
        A JSON-serialisable dictionary holding additional keyword parameters associated with the
        image. It is stored as-is, not copied. Default is an empty dictionary.

    Attributes
    ----------
    image : numpy.ndarray
        the pixel array
    pixel_format : str
        the pixel format
    meta : dict
        the metadata

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.image import Image
    >>> img = Image(np.zeros((2, 3, 3), dtype=np.uint8), meta={"id": 1})
    >>> img
    cv.Image(image.shape=(2, 3, 3), pixel_format='rgb', meta={"id": 1})
    >>> img2 = Image.from_json(img.to_json(image_codec="jpg"))
    >>> img2.image.shape, img2.pixel_format, img2.meta
    ((2, 3, 3), 'rgb', {'id': 1})
    """

    def __init__(self, image, pixel_format="rgb", meta={}):
        self.image = np.ascontiguousarray(image)  # need to be contiguous
        self.pixel_format = pixel_format
        self.meta = meta

    def __repr__(self):
        """Returns a short description of the image, its pixel format and its metadata."""
        return (
            f"cv.Image(image.shape={self.image.shape}, pixel_format='{self.pixel_format}', "
            f"meta={json.dumps(self.meta)})"
        )

    # ---- serialisation -----

    def to_json(self, image_codec: str = "jpg", quality: tp.Optional[int] = None):
        """Dumps the image to a JSON-like object.

        The pixels are encoded with the given codec and then base64-encoded. If the pixel format has
        an
        alpha channel, it is encoded separately in key 'alpha'.

        Parameters
        ----------
        image_codec : {'jpg', 'png'}, optional
            image codec. Only 'jpg' is currently implemented for this method. Default is 'jpg'.
        quality : int, optional
            percentage of image quality. For 'jpg', it is a value between 0 and 100. For 'png', it
            is a value between 0 and 9. If not provided, the backend default will be used.

        Returns
        -------
        json_obj : dict
            the serialised json object, with keys 'pixel_format', 'height', 'width', 'image_codec',
            'image_codec_quality' (only if `quality` is provided), 'meta', 'image' and, if the pixel
            format has an alpha channel, 'alpha'

        Raises
        ------
        NotImplementedError
            if `image_codec` is 'png'
        ValueError
            if `image_codec` is neither 'jpg' nor 'png'

        See Also
        --------
        from_json : the inverse operation

        Examples
        --------
        >>> import numpy as np
        >>> from mt.opencv.image import Image
        >>> img = Image(np.zeros((2, 3, 3), dtype=np.uint8), meta={"id": 1})
        >>> obj = img.to_json()
        >>> sorted(obj)
        ['height', 'image', 'image_codec', 'meta', 'pixel_format', 'width']
        >>> obj["height"], obj["width"], obj["pixel_format"], obj["image_codec"]
        (2, 3, 'rgb', 'jpg')
        """

        # meta
        json_obj = {}
        json_obj["pixel_format"] = self.pixel_format
        json_obj["height"] = self.image.shape[0]
        json_obj["width"] = self.image.shape[1]
        json_obj["image_codec"] = image_codec
        if quality is not None:
            json_obj["image_codec_quality"] = quality
        json_obj["meta"] = self.meta

        # image
        if image_codec == "jpg":
            img_bytes = _encode_jpeg(
                _to_cv_image(self.image, self.pixel_format), quality
            )
        elif image_codec == "png":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown image codec '{image_codec}'.")
        encoded = base64.b64encode(img_bytes)
        json_obj["image"] = encoded.decode("ascii")

        if self.pixel_format != "gray":
            a_id = _alpha_channel_index(self.pixel_format)
            if a_id >= 0:  # has alpha channel
                alpha_image = np.ascontiguousarray(self.image[:, :, a_id : a_id + 1])
                img_bytes = _encode_jpeg(alpha_image, quality)
                encoded = base64.b64encode(img_bytes)
                json_obj["alpha"] = encoded.decode("ascii")

        return json_obj

    def to_hdf5(
        self, h5_group, image_codec: str = "jpg", quality: tp.Optional[int] = None
    ):
        """Dumps the image to a h5py.Group object.

        The pixel format, resolution, codec and metadata are saved as attributes of the group. The
        encoded pixels are saved in dataset 'image', and for 'jpg' images with an alpha channel, the
        encoded alpha channel is saved in dataset 'alpha'.

        Parameters
        ----------
        h5_group : h5py.Group
            a :class:`h5py.Group` object to write to
        image_codec : {'jpg', 'png'}, optional
            image codec. Currently only 'jpg' and 'png' are supported. Default is 'jpg'.
        quality : int, optional
            percentage of image quality. For 'jpg', it is a value between 0 and 100. For 'png', it
            is a value between 0 and 9. If not provided, the backend default will be used.

        Raises
        ------
        ImportError
            if h5py is not importable
        ValueError
            if the provided group is not of type :class:`h5py.Group`, or if the codec is unknown
        RuntimeError
            if OpenCV fails to encode the image

        See Also
        --------
        from_hdf5 : the inverse operation
        """

        if not base.is_h5group(h5_group):
            raise ValueError("The provided group is not a h5py.Group instance.")

        h5_group.attrs["pixel_format"] = self.pixel_format
        h5_group.attrs["height"] = self.image.shape[0]
        h5_group.attrs["width"] = self.image.shape[1]
        h5_group.attrs["image_codec"] = image_codec
        if quality is not None:
            h5_group.attrs["image_codec_quality"] = quality
        h5_group.attrs["meta"] = json.dumps(self.meta)

        # image
        if image_codec == "jpg":
            img_bytes = _encode_jpeg(
                _to_cv_image(self.image, self.pixel_format), quality
            )
            h5_group.create_dataset(
                "image",
                data=np.frombytes(img_bytes),
                compression="gzip",
            )
        elif image_codec == "png":
            if quality is None:
                retval, x = cv2.imencode(".png", self.image)
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
                retval, x = cv2.imencode(".png", self.image, params)
            if not retval:
                raise RuntimeError(
                    f"Unable to use OpenCV to png-encode the image of shape {image.shape}."
                )
            h5_group["image"] = x
        else:
            raise ValueError(f"Unknown image codec '{image_codec}'.")

        if image_codec == "jpg" and self.pixel_format != "gray":
            a_id = _alpha_channel_index(self.pixel_format)
            if a_id >= 0:  # has alpha channel
                alpha_image = np.ascontiguousarray(self.image[:, :, a_id : a_id + 1])
                img_bytes = _encode_jpeg(alpha_image, quality)
                h5_group.create_dataset(
                    "alpha",
                    data=np.frombytes(img_bytes),
                    compression="gzip",
                )

    @staticmethod
    def from_json(json_obj):
        """Loads the image from a JSON-like object produced by :func:`Image.to_json`.

        Parameters
        ----------
        json_obj : dict
            the serialised json object. Keys 'pixel_format', 'meta' and 'image' are required, and so
            is
            'alpha' if the pixel format has an alpha channel.

        Returns
        -------
        Image
            the loaded image with metadata. The alpha channel, if any, is restored from key 'alpha'.

        Raises
        ------
        RuntimeError
            if the image bytes cannot be decoded

        See Also
        --------
        to_json : the inverse operation

        Examples
        --------
        >>> import numpy as np
        >>> from mt.opencv.image import Image
        >>> img = Image(np.full((2, 2), 7, dtype=np.uint8), pixel_format="gray")
        >>> Image.from_json(img.to_json()).image
        array([[7, 7],
               [7, 7]], dtype=uint8)
        """

        # meta
        pixel_format = json_obj["pixel_format"]
        image_codec = json_obj.get("image_codec", "jpg")
        meta = json_obj["meta"]

        decoded = base64.b64decode(json_obj["image"])
        if pixel_format == "gray":
            image = _decode_jpeg(decoded, gray=True)
        else:
            image = _from_cv_image(_decode_jpeg(decoded, gray=False), pixel_format)

        if pixel_format != "gray":
            a_id = _alpha_channel_index(pixel_format)
            if a_id >= 0:  # has alpha channel
                decoded = base64.b64decode(json_obj["alpha"])
                alpha_image = _decode_jpeg(decoded, gray=True)
                if len(alpha_image.shape) == 2:
                    alpha_image = alpha_image[:, :, None]
                image[:, :, a_id : a_id + 1] = alpha_image

        return Image(image, pixel_format=pixel_format, meta=meta)

    @staticmethod
    def from_hdf5(h5_group):
        """Loads the image from an HDF5 group written by :func:`Image.to_hdf5`.

        Parameters
        ----------
        h5_group : h5py.Group
            a :class:`h5py.Group` object to read from

        Returns
        -------
        Image
            the loaded image with metadata

        Raises
        ------
        ValueError
            if the provided group is not of type :class:`h5py.Group`
        RuntimeError
            if the image bytes cannot be decoded

        See Also
        --------
        to_hdf5 : the inverse operation
        """

        if not base.is_h5group(h5_group):
            raise ValueError("The provided group is not a h5py.Group instance.")

        # meta
        pixel_format = h5_group.attrs["pixel_format"]
        image_codec = h5_group.attrs.get("image_codec", "jpg")
        meta = json.loads(h5_group.attrs["meta"])

        if image_codec == "jpg":
            if "image" in h5_group:  # dataset?
                decoded = h5_group["image"][:].tobytes()
            else:  # attribute?
                decoded = h5_group.attrs["image"].tobytes()
            if pixel_format == "gray":
                image = _decode_jpeg(decoded, gray=True)
            else:
                image = _from_cv_image(_decode_jpeg(decoded, gray=False), pixel_format)

            if pixel_format != "gray":
                a_id = _alpha_channel_index(pixel_format)
                if a_id >= 0:  # has alpha channel
                    if "alpha" in h5_group:
                        decoded = h5_group["alpha"][:].tobytes()
                    else:
                        decoded = h5_group.attrs["alpha"].tobytes()
                    alpha_image = _decode_jpeg(decoded, gray=True)
                    if len(alpha_image.shape) == 2:
                        alpha_image = alpha_image[:, :, None]
                    image[:, :, a_id : a_id + 1] = alpha_image
        else:  # png
            image = cv2.imdecode(h5_group["image"][:], cv2.IMREAD_UNCHANGED)

        return Image(image, pixel_format=pixel_format, meta=meta)


async def immload_asyn(fp, context_vars: dict = {}):
    """An asyn function that loads an image with metadata.

    The file is first tried as HDF5 (if h5py is installed) and then as JSON.

    Parameters
    ----------
    fp : str or file-like object
        string representing a local filepath or an open readable file-like object. Only a filepath
        is supported for HDF5 files.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. Otherwise a
        :class:`KeyError` is raised when the file is accessed.

    Returns
    -------
    Image
        the loaded image with metadata

    Raises
    ------
    OSError
        if an error occured while loading, for example the file is neither valid HDF5 nor valid JSON

    Notes
    -----
    As of 2022/06/18, the file can be in HDF5 format or JSON format.

    See Also
    --------
    immload : the synchronous version
    immsave_asyn : the inverse operation
    """

    # try with h5py
    try:
        import h5py

        try:
            f = h5py.File(fp, "r")
            return Image.from_hdf5(f)
        except OSError:
            pass
    except ImportError:
        pass

    # try with json
    if not isinstance(fp, str):
        return Image.from_json(json.load(fp))
    try:
        json_obj = await aio.json_load(fp, context_vars=context_vars)
    except json.decoder.JSONDecodeError:
        if isinstance(fp, str):
            raise OSError(
                f"Unable to json-load filepath '{fp}'. It may be corrupted."
            )
        else:
            raise OSError("Unable to json-load. The file may be corrupted.")
    return Image.from_json(json_obj)


def immload(fp):
    """Loads an image with metadata.

    Parameters
    ----------
    fp : str or file-like object
        string representing a local filepath or an open readable file handle

    Returns
    -------
    Image
        the loaded image with metadata

    Raises
    ------
    OSError
        if an error occured while loading

    See Also
    --------
    immload_asyn : the asynchronous version
    immsave : the inverse operation

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mt.opencv.image import Image, immsave, immload
    >>> path = os.path.join(tempfile.mkdtemp(), "x.json")
    >>> img = Image(np.zeros((2, 3, 3), dtype=np.uint8), meta={"id": 1})
    >>> _ = immsave(img, path, image_codec="jpg", file_format="json")
    >>> immload(path)
    cv.Image(image.shape=(2, 3, 3), pixel_format='rgb', meta={"id": 1})
    """
    return aio.srun(immload_asyn, fp)


async def immload_header_asyn(fp, context_vars: dict = {}):
    """An asyn function that loads the header of an image with metadata.

    Only the pixel format, resolution and metadata are read. The pixels are not decoded.

    Parameters
    ----------
    fp : str or file-like object
        string representing a local filepath or an open readable file-like object. Only a filepath
        is supported for HDF5 files.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. Otherwise a
        :class:`KeyError` is raised when the file is accessed.

    Returns
    -------
    dict
        a dictionary containing keys `['pixel_format', 'width', 'height', 'meta']`

    Raises
    ------
    OSError
        if an error occured while loading

    Notes
    -----
    As of 2022/06/18, the file can be in HDF5 format or JSON format.

    See Also
    --------
    immload_header : the synchronous version
    """

    # try with h5py
    try:
        import h5py

        try:
            f = h5py.File(fp, "r")
            res = {
                "pixel_format": f.attrs["pixel_format"],
                "width": f.attrs["width"],
                "height": f.attrs["height"],
                "meta": json.loads(f.attrs["meta"]),
            }
            return res
        except OSError:
            pass
    except ImportError:
        pass

    # try with json
    if not isinstance(fp, str):
        json_obj = json.load(fp)
    else:
        try:
            json_obj = await aio.json_load(fp, context_vars=context_vars)
        except json.decoder.JSONDecodeError:
            if isinstance(fp, str):
                raise OSError(
                    f"Unable to json-load filepath '{fp}'. It may be corrupted."
                )
            else:
                raise OSError("Unable to json-load. The file may be corrupted.")

    res = {
        "pixel_format": json_obj["pixel_format"],
        "width": json_obj["width"],
        "height": json_obj["height"],
        "meta": json_obj["meta"],
    }
    return res


def immload_header(fp):
    """Loads the header of an image with metadata.

    Only the pixel format, resolution and metadata are read. The pixels are not decoded.

    Parameters
    ----------
    fp : str or file-like object
        string representing a local filepath or an open readable file-like object

    Returns
    -------
    dict
        a dictionary containing keys `['pixel_format', 'width', 'height', 'meta']`

    Raises
    ------
    OSError
        if an error occured while loading

    Notes
    -----
    As of 2022/06/18, the file can be in HDF5 format or JSON format. Unlike
    :func:`immload_header_asyn`, this function has no `context_vars` argument.

    See Also
    --------
    immload_header_asyn : the asynchronous version

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mt.opencv.image import Image, immsave, immload_header
    >>> path = os.path.join(tempfile.mkdtemp(), "x.json")
    >>> img = Image(np.zeros((2, 3, 3), dtype=np.uint8), meta={"id": 1})
    >>> _ = immsave(img, path, image_codec="jpg", file_format="json")
    >>> immload_header(path)
    {'pixel_format': 'rgb', 'width': 3, 'height': 2, 'meta': {'id': 1}}
    """
    return aio.srun(immload_header_asyn, fp)


async def immsave_asyn(
    image: Image,
    fp: str,
    file_mode: int = 0o664,
    image_codec: str = "png",
    quality: tp.Optional[int] = None,
    context_vars: dict = {},
    file_format: str = "hdf5",
    file_write_delayed: bool = False,
    make_dirs: bool = False,
    logger=None,
):
    """An asyn function that saves an image with metadata to file.

    Parameters
    ----------
    image : Image
        an image with metadata
    fp : str or file-like object
        local filepath to save the content to. If the file format is 'json', fp can also be a
        file-like object.
    file_mode : int, optional
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen. Default is 0o664.
    image_codec : {'jpg', 'png'}, optional
        image codec. Default is 'png'. Note that file format 'json' only supports 'jpg' at the
        moment, so for that format the codec must be given explicitly.
    quality : int, optional
        percentage of image quality. For 'jpg', it is a value between 0 and 100. For 'png', it is
        a value between 0 and 9. If not provided, the backend default will be used.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. Otherwise a
        :class:`KeyError` is raised when the file is accessed.
    file_format : {'json', 'hdf5'}, optional
        format to be used for saving the content. Default is 'hdf5', which requires h5py.
    file_write_delayed : bool, optional
        Only valid in asynchronous mode and the format is 'json'. If True, wraps the file write
        task into a future and returns the future. In all other cases, proceeds as usual.
    make_dirs : bool, optional
        Whether or not to make the folders containing the path before writing to the file. Only used
        for the 'json' format with a filepath.
    logger : logging.Logger, optional
        logger for debugging purposes

    Returns
    -------
    asyncio.Future or object
        In the case of format 'json', it is either a future or whatever :func:`json.dump` returns,
        depending on whether the file write task is delayed or not. In the case format 'hdf5', it
        is whatever :func:`Image.to_hdf5` returns.

    Raises
    ------
    ImportError
        if the format is 'hdf5' but h5py cannot be imported
    ValueError
        if the file format is unknown, or the format is 'hdf5' and `fp` is not a string
    NotImplementedError
        if the format is 'json' and the codec is 'png'
    OSError
        if an error occured while saving

    See Also
    --------
    immsave : the synchronous version
    immload_asyn : the inverse operation
    """

    if file_format == "hdf5":
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "Unable to import h5py. You need to pip install it for "
                "mt.opencv.Image to save to HDF5 format."
            )

        if not isinstance(fp, str):
            raise ValueError(
                f"For hdf5 format, argument 'fp' must be a string. Got: {type(fp)}."
            )

        async with aio.CreateFileH5(
            fp, file_mode=file_mode, context_vars=context_vars, logger=logger
        ) as h5file:
            retval = image.to_hdf5(
                h5file.handle, image_codec=image_codec, quality=quality
            )
    elif file_format == "json":
        json_obj = image.to_json(image_codec=image_codec, quality=quality)

        if isinstance(fp, str):
            retval = await aio.json_save(
                fp,
                json_obj,
                indent=4,
                file_mode=file_mode,
                context_vars=context_vars,
                file_write_delayed=file_write_delayed,
                make_dirs=make_dirs,
            )
        else:
            retval = json.dump(json_obj, fp, indent=4)
    else:
        raise ValueError(f"Unnkown file format '{file_format}'.")

    return retval


def immsave(
    image,
    fp,
    file_mode: int = 0o664,
    image_codec: str = "png",
    quality: tp.Optional[int] = None,
    file_format: str = "hdf5",
    make_dirs: bool = False,
    logger=None,
):
    """Saves an image with metadata to file.

    Parameters
    ----------
    image : Image
        an image with metadata
    fp : str or file-like object
        string representing a local filepath or an open writable file handle. A file handle is only
        supported for the 'json' format.
    file_mode : int, optional
        file mode to be set to using :func:`os.chmod`. Only valid if fp is a string. If None is
        given, no setting of file mode will happen. Default is 0o664.
    image_codec : {'jpg', 'png'}, optional
        image codec. Default is 'png'. Note that file format 'json' only supports 'jpg' at the
        moment, so for that format the codec must be given explicitly.
    quality : int, optional
        percentage of image quality. For 'jpg', it is a value between 0 and 100. For 'png', it is
        a value between 0 and 9. If not provided, the backend default will be used.
    file_format : {'json', 'hdf5'}, optional
        format to be used for saving the content. Default is 'hdf5', which requires h5py.
    make_dirs : bool, optional
        Whether or not to make the folders containing the path before writing to the file.
    logger : logging.Logger, optional
        logger for debugging purposes

    Raises
    ------
    ImportError
        if the format is 'hdf5' but h5py cannot be imported
    ValueError
        if the file format is unknown
    NotImplementedError
        if the format is 'json' and the codec is 'png'
    OSError
        if an error occured while saving

    See Also
    --------
    immsave_asyn : the asynchronous version
    immload : the inverse operation

    Examples
    --------
    See :func:`immload`.
    """
    return aio.srun(
        immsave_asyn,
        image,
        fp,
        file_mode=file_mode,
        image_codec=image_codec,
        quality=quality,
        file_format=file_format,
        make_dirs=make_dirs,
        logger=logger,
    )


async def imload(
    filepath: str,
    flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
    context_vars: dict = {},
):
    """An asyn function wrapping on :func:`cv2.imread`.

    The file is read as bytes and then decoded by :func:`cv2.imdecode`, so the path can be on any
    filesystem supported by :mod:`mt.aio`. Channels of a colour image are in BGR order, as OpenCV
    does.

    Parameters
    ----------
    filepath : str
        Local path to the file to be loaded
    flags : int, optional
        'cv.IMREAD_xxx' flags, if any. See :func:`cv2.imread`. Default is
        `cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH`.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. Otherwise a
        :class:`KeyError` is raised when the file is accessed.

    Returns
    -------
    img : numpy.ndarray
        the loaded image, of shape `(height, width)` or `(height, width, nchannels)`. It is None if
        the content cannot be decoded.

    See Also
    --------
    cv2.imread
        wrapped function
    imsave : the inverse operation

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from mt import aio
    >>> from mt.opencv.image import imload, imsave
    >>> path = os.path.join(tempfile.mkdtemp(), "x.png")
    >>> img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    >>> _ = aio.srun(imsave, path, img)
    >>> aio.srun(imload, path)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=uint8)
    """

    contents = await aio.read_binary(filepath, context_vars=context_vars)
    buf = np.asarray(bytearray(contents), dtype=np.uint8)
    return cv2.imdecode(buf, flags=flags)


async def imsave(
    filepath: str,
    img: np.ndarray,
    params=None,
    file_mode: int = 0o664,
    context_vars: dict = {},
    file_write_delayed: bool = False,
    make_dirs: bool = False,
):
    """An asyn function wrapping on :func:`cv2.imwrite`.

    The image is encoded by :func:`cv2.imencode` according to the extension of `filepath`, and the
    bytes are then written to the file.

    Parameters
    ----------
    filepath : str
        Local path to the file to be saved to. Its extension determines the image format.
    img : numpy.ndarray
        the image to be saved. A colour image is assumed by OpenCV to be in BGR(A) order.
    params : list, optional
        Format-specific parameters, if any, as a flat list of `[flag, value, ...]`, like those
        'cv.IMWRITE_xxx' flags. See :func:`cv2.imencode`.
    file_mode : int, optional
        file mode to be set to using :func:`os.chmod`. Only valid if fp is a string. If None is
        given, no setting of file mode will happen. Default is 0o664.
    context_vars : dict, optional
        context variables within which the function runs. It must include `context_vars['async']`
        (bool), telling whether to invoke the function asynchronously or not. Otherwise a
        :class:`KeyError` is raised when the file is accessed.
    file_write_delayed : bool, optional
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual.
    make_dirs : bool, optional
        Whether or not to make the folders containing the path before writing to the file.

    Returns
    -------
    asyncio.Future or int
        either a future or the number of bytes written, depending on whether the file write
        task is delayed or not

    Raises
    ------
    ValueError
        if the image cannot be encoded

    Note
    ----
    Do not use this function to write in PNG format. OpenCV would happily assume the input is BGR
    or BGRA and then write to PNG under that assumption, which often results in a wrong order.

    See Also
    --------
    cv2.imwrite
        wrapped function
    imload : the inverse operation

    Examples
    --------
    See :func:`imload`.
    """

    ext = path.splitext(filepath)[1]
    res, contents = cv2.imencode(ext, img, params=params)

    if res is not True:
        raise ValueError("Unable to encode the input image.")

    buf = np.array(contents.tostring())
    return await aio.write_binary(
        filepath,
        buf,
        file_mode=file_mode,
        context_vars=context_vars,
        file_write_delayed=file_write_delayed,
        make_dirs=make_dirs,
    )


def im_float2ubyte(img: np.ndarray, is_float01: bool = True):
    """Converts an image with a float dtype into an image with an ubyte dtype.

    Values are scaled and rounded to the nearest integer. Values outside of the expected range
    are not clipped and will wrap around.

    Parameters
    ----------
    img : numpy.ndarray
        the image to be converted
    is_float01 : bool, optional
        whether the pixel values of the float image are in range [0,1] (True) or range [-1,1]
        (False). Default is True.

    Returns
    -------
    numpy.ndarray
        the converted image with ubyte dtype, with the same shape as `img`

    See Also
    --------
    im_ubyte2float : the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.image import im_float2ubyte
    >>> im_float2ubyte(np.array([0.0, 0.5, 1.0]))
    array([  0, 128, 255], dtype=uint8)
    >>> im_float2ubyte(np.array([-1.0, 0.0, 1.0]), is_float01=False)
    array([  0, 128, 255], dtype=uint8)
    """
    if is_float01:
        return np.round(img * 255.0).astype(np.uint8)
    return np.round((img * 127.5) + 127.5).astype(np.uint8)


def im_ubyte2float(
    img: np.ndarray,
    is_float01: bool = True,
    rng: tp.Union[np.random.RandomState, bool, None] = None,
):
    """Converts an image with an ubyte dtype into an image with a float32 dtype.

    Parameters
    ----------
    img : numpy.ndarray
        the image to be converted
    is_float01 : bool, optional
        whether the pixel values of the float image are to be in range [0,1] (True) or range [-1,1]
        (False). Default is True.
    rng : numpy.random.RandomState or bool or None, optional
        Whether or not to use an rng to dequantise pixel values from integer to floats. If None or
        False is provided, we do not add (0,1)-uniform noise to the pixel values. If True is
        provided, an internal 'rng' is created. Otherwise, the provided 'rng' is used to generate
        random numbers. When dequantising, the divisor is 256 (or 128) instead of 255 (or 127.5).

    Returns
    -------
    numpy.ndarray
        the converted image with float32 dtype, with the same shape as `img`

    See Also
    --------
    im_float2ubyte : the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.image import im_ubyte2float
    >>> im_ubyte2float(np.array([0, 51, 255], dtype=np.uint8))
    array([0. , 0.2, 1. ], dtype=float32)
    >>> im_ubyte2float(np.array([0, 255], dtype=np.uint8), is_float01=False)
    array([-1.,  1.], dtype=float32)
    """
    if rng is True:
        rng = np.random.RandomState()
    if isinstance(rng, np.random.RandomState):
        img = np.dequantise_images(img, rng)
        if is_float01:
            return (img / 256.0).astype(np.float32)
        return ((img / 128.0) - 1).astype(np.float32)

    if is_float01:
        return (img / 255.0).astype(np.float32)
    return ((img / 127.5) - 1).astype(np.float32)
