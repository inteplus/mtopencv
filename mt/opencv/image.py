'''An self-contained image.'''


import aiofiles
import base64
import json
import turbojpeg as tj
_tj = tj.TurboJPEG()
import cv2

from mt import np
from mt.base import path, aio


__all__ = ['PixelFormat', 'Image', 'immload_asyn', 'immload', 'immsave_asyn', 'immsave', 'imload', 'imsave', 'im_float2ubyte', 'im_ubyte2float']



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
        self.image = np.ascontiguousarray(image) # need to be contiguous
        self.pixel_format = pixel_format
        self.meta = meta

    def __repr__(self):
        return "cv.Image(image.shape={}, pixel_format='{}', meta={})".format(self.image.shape, self.pixel_format, json.dumps(self.meta))

    def check(self):
        '''Checks for data consistency, raising a ValueError if something has gone wrong.'''
        if len(self.image.shape) == 2:
            if self.pixel_format != 'gray':
                raise ValueError("Pixel format is not 'gray' but image has only one channel.")
            self.image = self.image.reshape(self.image.shape + (1,))
        elif len(self.image.shape) == 3:
            desired_nchannels = PixelFormat[self.pixel_format][1]
            if self.image.shape[2] != desired_nchannels:
                raise ValueError("Unexpected number of channels {}. It should be {} for pixel format '{}'.".format(self.image.shape[2], desired_nchannels, self.pixel_format))
        else:
            raise ValueError("Unexpected image shape {}. It must have 2 or 3 dimensions.".format(self.image.shape))

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
        img_bytes = _tj.encode(self.image, quality=quality, pixel_format=tj_params[0], jpeg_subsample=tj_params[2])
        encoded = base64.b64encode(img_bytes)
        json_obj['image'] = encoded.decode('ascii')

        if self.pixel_format != 'gray':
            a_id = self.pixel_format.find('a')
            if a_id >= 0: # has alpha channel
                alpha_image = np.ascontiguousarray(self.image[:,:,a_id:a_id+1])
                img_bytes = _tj.encode(alpha_image, quality=quality, pixel_format=tj.TJPF_GRAY, jpeg_subsample=tj.TJSAMP_GRAY)
                encoded = base64.b64encode(img_bytes)
                json_obj['alpha'] = encoded.decode('ascii')

        return json_obj

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
            if a_id >= 0: # has alpha channel
                decoded = base64.b64decode(json_obj['alpha'])
                alpha_image = _tj.decode(decoded, pixel_format=tj.TJPF_GRAY)
                image[:,:,a_id:a_id+1] = alpha_image

        return Image(image, pixel_format=pixel_format, meta=meta)


async def immload_asyn(fp, context_vars: dict = {}):
    '''An asyn function that loads an image with metadata.

    Parameters
    ----------
    fp : object
        string representing a local filepath or an open readable file handle
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
    '''

    if not isinstance(fp, str):
        return Image.from_json(json.load(fp))
    json_obj = await aio.json_load(fp, context_vars=context_vars)
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


async def immsave_asyn(image, fp, file_mode: int = 0o664, quality: float = 90, context_vars: dict = {}, file_write_delayed: bool = False):
    '''An asyn function that saves an image with metadata to file.

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
    context_vars : dict
        a dictionary of context variables within which the function runs. It must include
        `context_vars['async']` to tell whether to invoke the function asynchronously or not.
    file_write_delayed : bool
        Only valid in asynchronous mode. If True, wraps the file write task into a future and
        returns the future. In all other cases, proceeds as usual.

    Returns
    -------
    asyncio.Future or object
        either a future or whatever :func:`json.dump` returns, depending on whether the file write
        task is delayed or not

    Raises
    ------
    OSError
        if an error occured while loading
    '''

    json_obj = image.to_json(quality=quality)

    if isinstance(fp, str):
        retval = await aio.json_save(fp, json_obj, indent=4, file_mode=file_mode, context_vars=context_vars, file_write_delayed=file_write_delayed)
    else:
        retval = json.dump(json_obj, fp, indent=4)

    return retval


def immsave(image, fp, file_mode: int = 0o664, quality: float = 90):
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

    Raises
    ------
    OSError
        if an error occured while loading
    '''
    return aio.srun(immsave_asyn, image, fp, file_mode=file_mode, quality=quality)


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
