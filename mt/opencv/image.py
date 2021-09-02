'''An self-contained image.'''


import asyncio
import aiofiles
import base64
import json
import turbojpeg as tj
_tj = tj.TurboJPEG()

from mt import np
from mt.base import path
from mt.base.deprecated import deprecated_func


__all__ = ['PixelFormat', 'Image', 'immload', 'immsave', 'immload_aio', 'immsave_aio', 'imload', 'imsave']



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


@deprecated_func("0.4", suggested_func="immload_aio", removed_version="1.0", docstring_prefix="    ")
def immload(fp):
    '''Loads an image with metadata in the usual IO-blocking way.

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

    return asyncio.run(immload_aio(fp))


async def immload_aio(fp):
    '''Loads an image with metadata asynchronously.

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

    if not isinstance(fp, str):
        return Image.from_json(json.load(fp))

    async with aiofiles.open(fp, mode='rt') as f:
        contents = await f.read()
        return Image.from_json(json.loads(contents))


@deprecated_func("0.4", suggested_func="immload_aio", removed_version="1.0", docstring_prefix="    ")
def immsave(image, fp, file_mode=0o664, quality=90):
    '''Saves an image with metadata to file in the usual IO-blocking way.

    Parameters
    ----------
    imm : Image
        an image with metadata
    fp : object
        string representing a local filepath or an open writable file handle
    file_mode : int
        file mode to be set to using :func:`os.chmod`. Only valid if fp is a string. If None is given, no setting of file mode will happen.
    quality : int
        percentage of image quality. Default is 90.

    Raises
    ------
    OSError
        if an error occured while loading
    '''

    return asyncio.run(immsave_aio(image, fp, file_mode=file_mode, quality=quality))


async def immsave_aio(image, fp, file_mode=0o664, quality=90):
    '''Saves an image with metadata to file asynchronously.

    Parameters
    ----------
    imm : Image
        an image with metadata
    fp : object
        string representing a local filepath or an open writable file handle
    file_mode : int
        file mode to be set to using :func:`os.chmod`. Only valid if fp is a string. If None is given, no setting of file mode will happen.
    quality : int
        percentage of image quality. Default is 90.

    Raises
    ------
    OSError
        if an error occured while loading
    '''

    if not isinstance(fp, str):
        return json.dump(image.to_json(quality=quality), fp, indent=4)

    async with aiofiles.open(fp, mode='wt') as fp2:
        contents = json.dumps(image.to_json(quality=quality), fp2, indent=4)
        await f.write(contents)
    if file_mode:  # chmod
        path.chmod(fp, file_mode)


async def imload(filepath: str, flags=None):
    '''Wrapper on :func:`cv.imread` but with asynchronous IO.

    Parameters
    ----------
    filepath : str
        Local path to the file to be loaded
    flags : int
        'cv.IMREAD_xxx' flags, if any. See :func:`cv:imread`.

    Returns
    -------
    img : numpy.ndarray
        the loaded image

    See Also
    --------
    cv.imread
        wrapped function
    '''

    async with aiofiles.open(filepath, mode='rb') as f:
        contents = await f.read()

    from cv2 import imdecode
    buf = np.asarray(bytearray(contents), dtype=np.uint8)
    return imdecode(buf, flags=flags)


async def imsave(filepath: str, img: np.ndarray, params=None):
    '''Wrapper on :func:`cv.imwrite` but with asynchronous IO.

    Parameters
    ----------
    filepath : str
        Local path to the file to be saved to
    img : numpy.ndarray
        the image to be saved
    params : int
        Format-specific parameters, if any. Like those 'cv.IMWRITE_xxx' flags. See :func:`cv.imwrite`.

    See Also
    --------
    cv.imwrite
        wrapped function
    '''

    ext = path.splitext(filepath)[1]
    from cv2 import imencode
    res, contents = imencode(ext, img, params=params)

    if res is not True:
        raise ValueError("Unable to encode the input image.")

    buf = np.array(contents.tostring())
    async with aiofiles.open(filepath, mode='wb') as f:
        await f.write(buf)
