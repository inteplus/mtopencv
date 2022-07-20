'''An self-contained image.'''


from mt.base import path, aio
from mt import np
import cv2
import aiofiles
import base64
import json
import turbojpeg as tj
_tj = tj.TurboJPEG()


__all__ = ['PixelFormat', 'Image', 'immload_asyn', 'immload', 'immload_header_asyn',
           'immload_header', 'immsave_asyn', 'immsave', 'imload', 'imsave', 'im_float2ubyte',
           'im_ubyte2float']


PixelFormat = {
    'rgb': (tj.TJPF_RGB, 3, tj.TJSAMP_422),
    'bgr': (tj.TJPF_BGR, 3, tj.TJSAMP_422),
    'rgba': (tj.TJPF_RGBA, 4, tj.TJSAMP_422),
    'bgra': (tj.TJPF_BGRA, 4, tj.TJSAMP_422),
    'argb': (tj.TJPF_ARGB, 4, tj.TJSAMP_422),
    'abgr': (tj.TJPF_ABGR, 4, tj.TJSAMP_422),
    'gray': (tj.TJPF_GRAY, 1, tj.TJSAMP_GRAY),
}


class Image(object):
    '''A self-contained image, where the meta-data associated with the image are kept together with the image itself.

    Parameters
    ----------
    image : numpy.array
        a 2D image of shape (height, width, nchannels) or (height, width) with dtype uint8
    pixel_format : str
        one of the keys in the PixelFormat mapping
    meta : dict
        A JSON-like object. It holds additional keyword parameters associated with the image.
    '''

    def __init__(self, image, pixel_format='rgb', meta={}):
        self.image = np.ascontiguousarray(image)  # need to be contiguous
        self.pixel_format = pixel_format
        self.meta = meta

    def __repr__(self):
        return "cv.Image(image.shape={}, pixel_format='{}', meta={})".format(self.image.shape, self.pixel_format, json.dumps(self.meta))

    def check(self):
        '''Checks for data consistency, raising a ValueError if something has gone wrong.'''
        if len(self.image.shape) == 2:
            if self.pixel_format != 'gray':
                raise ValueError(
                    "Pixel format is not 'gray' but image has only one channel.")
            self.image = self.image.reshape(self.image.shape + (1,))
        elif len(self.image.shape) == 3:
            desired_nchannels = PixelFormat[self.pixel_format][1]
            if self.image.shape[2] != desired_nchannels:
                raise ValueError("Unexpected number of channels {}. It should be {} for pixel format '{}'.".format(
                    self.image.shape[2], desired_nchannels, self.pixel_format))
        else:
            raise ValueError(
                "Unexpected image shape {}. It must have 2 or 3 dimensions.".format(self.image.shape))

    # ---- serialisation -----

    def to_json(self, quality=90):
        '''Dumps the image to a JSON-like object.

        Parameters
        ----------
        quality : int
            percentage of image quality. Default is 90.

        Returns
        -------
        json_obj : dict
            the serialised json object
        '''

        # meta
        json_obj = {}
        json_obj['pixel_format'] = self.pixel_format
        json_obj['height'] = self.image.shape[0]
        json_obj['width'] = self.image.shape[1]
        json_obj['meta'] = self.meta

        # image
        tj_params = PixelFormat[self.pixel_format]
        img_bytes = _tj.encode(self.image, quality=quality,
                               pixel_format=tj_params[0], jpeg_subsample=tj_params[2])
        encoded = base64.b64encode(img_bytes)
        json_obj['image'] = encoded.decode('ascii')

        if self.pixel_format != 'gray':
            a_id = self.pixel_format.find('a')
            if a_id >= 0:  # has alpha channel
                alpha_image = np.ascontiguousarray(
                    self.image[:, :, a_id:a_id+1])
                img_bytes = _tj.encode(
                    alpha_image, quality=quality, pixel_format=tj.TJPF_GRAY, jpeg_subsample=tj.TJSAMP_GRAY)
                encoded = base64.b64encode(img_bytes)
                json_obj['alpha'] = encoded.decode('ascii')

        return json_obj

    def to_hdf5(self, h5_group, quality=90):
        '''Dumps the image to a h5py.Group object.

        Parameters
        ----------
        h5_group : h5py.Group
            a :class:`h5py.Group` object to write to
        quality : int
            percentage of image quality. Default is 90.

        Raises
        ------
        ImportError
            if h5py is not importable
        ValueError
            if the provided group is not of type :class:`h5py.Group`
        '''

        import h5py

        if not isinstance(h5_group, h5py.Group):
            raise ValueError(
                "The provided group is not a h5py.Group instance.")

        h5_group.attrs['pixel_format'] = self.pixel_format
        h5_group.attrs['height'] = self.image.shape[0]
        h5_group.attrs['width'] = self.image.shape[1]
        h5_group.attrs['meta'] = json.dumps(self.meta)

        # image
        tj_params = PixelFormat[self.pixel_format]
        img_bytes = _tj.encode(self.image, quality=quality,
                               pixel_format=tj_params[0], jpeg_subsample=tj_params[2])
        h5_group.attrs['image'] = np.void(img_bytes)

        if self.pixel_format != 'gray':
            a_id = self.pixel_format.find('a')
            if a_id >= 0:  # has alpha channel
                alpha_image = np.ascontiguousarray(
                    self.image[:, :, a_id:a_id+1])
                img_bytes = _tj.encode(
                    alpha_image, quality=quality, pixel_format=tj.TJPF_GRAY, jpeg_subsample=tj.TJSAMP_GRAY)
                h5_group.attrs['alpha'] = np.void(img_bytes)

    @staticmethod
    def from_json(json_obj):
        '''Loads the image from a JSON-like object produced by :func:`dumps`.

        Parameters
        ----------
        json_obj : dict
            the serialised json object

        Returns
        -------
        Image
            the loaded image with metadata
        '''

        # meta
        pixel_format = json_obj['pixel_format']
        meta = json_obj['meta']

        decoded = base64.b64decode(json_obj['image'])
        image = _tj.decode(decoded, pixel_format=PixelFormat[pixel_format][0])

        if pixel_format != 'gray':
            a_id = pixel_format.find('a')
            if a_id >= 0:  # has alpha channel
                decoded = base64.b64decode(json_obj['alpha'])
                alpha_image = _tj.decode(decoded, pixel_format=tj.TJPF_GRAY)
                image[:, :, a_id:a_id+1] = alpha_image

        return Image(image, pixel_format=pixel_format, meta=meta)

    @staticmethod
    def from_hdf5(h5_group):
        '''Loads the image from an HDF5 group.

        Parameters
        ----------
        h5_group : h5py.Group
            a :class:`h5py.Group` object to read from

        Returns
        -------
        Image
            the loaded image with metadata
        '''

        import h5py

        if not isinstance(h5_group, h5py.Group):
            raise ValueError(
                "The provided group is not a h5py.Group instance.")

        # meta
        pixel_format = h5_group.attrs['pixel_format']
        meta = json.loads(h5_group.attrs['meta'])

        decoded = h5_group.attrs['image'].tobytes()
        image = _tj.decode(decoded, pixel_format=PixelFormat[pixel_format][0])

        if pixel_format != 'gray':
            a_id = pixel_format.find('a')
            if a_id >= 0:  # has alpha channel
                decoded = h5_group.attrs['alpha'].tobytes()
                alpha_image = _tj.decode(decoded, pixel_format=tj.TJPF_GRAY)
                image[:, :, a_id:a_id+1] = alpha_image

        return Image(image, pixel_format=pixel_format, meta=meta)


async def immload_asyn(fp, context_vars: dict = {}):
    '''An asyn function that loads an image with metadata.

    Parameters
    ----------
    fp : object
        string representing a local filepath or an open readable file-like object
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    Image
        the loaded image with metadata

    Raises
    ------
    OSError
        if an error occured while loading

    Notes
    -----
    As of 2022/06/18, the file can be in HDF5 format or JSON format.
    '''

    # try with h5py
    try:
        import h5py
        try:
            f = h5py.File(fp, 'r')
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
                "Unable to json-load filepath '{}'. It may be corrupted.".format(fp))
        else:
            raise OSError("Unable to json-load. The file may be corrupted.")
    return Image.from_json(json_obj)


def immload(fp):
    '''Loads an image with metadata.

    Parameters
    ----------
    fp : object
        string representing a local filepath or an open readable file handle

    Returns
    -------
    Image
        the loaded image with metadata

    Raises
    ------
    OSError
        if an error occured while loading
    '''
    return aio.srun(immload_asyn, fp)


async def immload_header_asyn(fp, context_vars: dict = {}):
    '''An asyn function that loads the header of an image with metadata.

    Parameters
    ----------
    fp : object
        string representing a local filepath or an open readable file-like object
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    dict
        a dictionary containing keys `['pixel_format', 'width', 'height', meta']`

    Raises
    ------
    OSError
        if an error occured while loading

    Notes
    -----
    As of 2022/06/18, the file can be in HDF5 format or JSON format.
    '''

    # try with h5py
    try:
        import h5py
        try:
            f = h5py.File(fp, 'r')
            res = {
                'pixel_format': f.attrs['pixel_format'],
                'width': f.attrs['width'],
                'height': f.attrs['height'],
                'meta': json.loads(f.attrs['meta']),
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
                    "Unable to json-load filepath '{}'. It may be corrupted.".format(fp))
            else:
                raise OSError(
                    "Unable to json-load. The file may be corrupted.")

    res = {
        'pixel_format': json_obj['pixel_format'],
        'width': json_obj['width'],
        'height': json_obj['height'],
        'meta': json_obj['meta'],
    }
    return res


def immload_header(fp):
    '''Loads the header of an image with metadata.

    Parameters
    ----------
    fp : object
        string representing a local filepath or an open readable file-like object
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    dict
        a dictionary containing keys `['pixel_format', 'width', 'height', meta']`

    Raises
    ------
    OSError
        if an error occured while loading

    Notes
    -----
    As of 2022/06/18, the file can be in HDF5 format or JSON format.
    '''
    return aio.srun(immload_header_asyn, fp)


async def immsave_asyn(
        image: Image,
        fp: str,
        file_mode: int = 0o664,
        quality: float = 90,
        context_vars: dict = {},
        file_format: str = 'hdf5',
        file_write_delayed: bool = False,
        logger=None,
):
    '''An asyn function that saves an image with metadata to file.

    Parameters
    ----------
    imm : Image
        an image with metadata
    fp : str
        local filepath to save the content to. If the file format is 'json', fp can also be a
        file-like object.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. If None is given, no setting of file mode
        will happen.
    quality : float
        percentage of image quality. Default is 90.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    file_format : {'json', 'hdf5'}
        format to be used for saving the content.
    file_write_delayed : bool
        Only valid in asynchronous mode and the format is 'json'. If True, wraps the file write
        task into a future and returns the future. In all other cases, proceeds as usual.
    logger : logging.Logger, optional
        logger for debugging purposes

    Returns
    -------
    asyncio.Future or object
        In the case of format 'json', it is either a future or whatever :func:`json.dump` returns,
        depending on whether the file write task is delayed or not. In the case format 'hdf5', it
        is the file size in bytes.

    Raises
    ------
    OSError
        if an error occured while loading
    '''

    if file_format == 'hdf5':
        try:
            import h5py
        except ImportError:
            raise ImportError("Unable to import h5py. You need to pip install it for "
                              "mt.opencv.Image to save to HDF5 format.")

        if not isinstance(fp, str):
            raise ValueError("For hdf5 format, argument 'fp' must be a string. Got: {}.".format(type(fp)))

        async with aio.CreateFileH5(fp, file_mode=file_mode, context_vars=context_vars, logger=logger) as h5file:
            image.to_hdf5(h5file.handle, quality=quality)

        retval = path.getsize(fp)
    elif file_format == 'json':
        json_obj = image.to_json(quality=quality)

        if isinstance(fp, str):
            retval = await aio.json_save(fp, json_obj, indent=4, file_mode=file_mode, context_vars=context_vars, file_write_delayed=file_write_delayed)
        else:
            retval = json.dump(json_obj, fp, indent=4)
    else:
        raise ValueError("Unnkown file format '{}'.".format(file_format))

    return retval


def immsave(image, fp, file_mode: int = 0o664, quality: float = 90, file_format: str = 'hdf5', logger=None):
    '''Saves an image with metadata to file.

    Parameters
    ----------
    imm : Image
        an image with metadata
    fp : object
        string representing a local filepath or an open writable file handle
    file_mode : int
        file mode to be set to using :func:`os.chmod`. Only valid if fp is a string. If None is
        given, no setting of file mode will happen.
    quality : float
        percentage of image quality. Default is 90.
    file_format : {'json', 'hdf5'}
        format to be used for saving the content.
    logger : logging.Logger, optional
        logger for debugging purposes

    Raises
    ------
    OSError
        if an error occured while loading
    '''
    return aio.srun(
        immsave_asyn, image, fp, file_mode=file_mode, quality=quality, file_format=file_format, logger=logger)


async def imload(filepath: str, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH, context_vars: dict = {}):
    '''An asyn function wrapping on :func:`cv.imread`.

    Parameters
    ----------
    filepath : str
        Local path to the file to be loaded
    flags : int
        'cv.IMREAD_xxx' flags, if any. See :func:`cv:imread`.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.

    Returns
    -------
    img : numpy.ndarray
        the loaded image

    See Also
    --------
    cv.imread
        wrapped function
    '''

    contents = await aio.read_binary(filepath, context_vars=context_vars)
    buf = np.asarray(bytearray(contents), dtype=np.uint8)
    return cv2.imdecode(buf, flags=flags)


async def imsave(filepath: str, img: np.ndarray, params=None, file_mode: int = 0o664, context_vars: dict = {}, file_write_delayed: bool = False):
    '''An asyn function wrapping on :func:`cv.imwrite`.

    Parameters
    ----------
    filepath : str
        Local path to the file to be saved to
    img : numpy.ndarray
        the image to be saved
    params : int
        Format-specific parameters, if any. Like those 'cv.IMWRITE_xxx' flags. See :func:`cv.imwrite`.
    file_mode : int
        file mode to be set to using :func:`os.chmod`. Only valid if fp is a string. If None is
        given, no setting of file mode will happen.
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    file_write_delayed : bool
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual.

    Returns
    -------
    asyncio.Future or int
        either a future or the number of bytes written, depending on whether the file write
        task is delayed or not

    Note
    ----
    Do not use this function to write in PNG format. OpenCV would happily assume the input is BGR
    or BGRA and then write to PNG under that assumption, which often results in a wrong order.

    See Also
    --------
    cv.imwrite
        wrapped asynchronous function
    '''

    ext = path.splitext(filepath)[1]
    res, contents = cv2.imencode(ext, img, params=params)

    if res is not True:
        raise ValueError("Unable to encode the input image.")

    buf = np.array(contents.tostring())
    return await aio.write_binary(filepath, buf, file_mode=file_mode, context_vars=context_vars, file_write_delayed=file_write_delayed)


def im_float2ubyte(img: np.ndarray, is_float01=True):
    '''Converts an image with a float dtype into an image with an ubyte dtype.

    Parameters
    ----------
    img : nd.ndarray
        the image to be converted
    is_float01 : bool
        whether the pixel values of the float image are in range [0,1] (True) or range [-1,1] (False)

    Returns
    -------
    nd.ndarray
        the converted image with ubyte dtype
    '''
    if is_float01:
        return (img*255.0).astype(np.uint8)
    return ((img*127.5)+127.5).astype(np.uint8)


def im_ubyte2float(img: np.ndarray, is_float01=True):
    '''Converts an image with an ubyte dtype into an image with a float32 dtype.

    Parameters
    ----------
    img : nd.ndarray
        the image to be converted
    is_float01 : bool
        whether the pixel values of the float image are to be in range [0,1] (True) or range [-1,1] (False)

    Returns
    -------
    nd.ndarray
        the converted image with float32 dtype
    '''
    if is_float01:
        return (img/255.0).astype(np.float32)
    return ((img/127.5)-1).astype(np.float32)


# MT-TODO: implement Image.to_hdf5(), Image.from_hdf5() and adjust immload_asyn() and immsave_asyn() to deal with HDF5 files
