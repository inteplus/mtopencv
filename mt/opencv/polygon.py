'''Extra functions dealing with polygons via OpenCV.'''

from . import cv2 as _cv
import numpy as _np


__all__ = ['render_mask', 'polygon2mask', 'morph_open']


def render_mask(contours, out_imgres, thickness=-1, debug=False):
    '''Renders a mask array from a list of contours.

    Parameters
    ----------
    contours : list
        a list of numpy arrays, each of which is a list of 2D points, not necessarily in integers
    out_imgres : list
        the [width, height] image resolution of the output mask.
    thickness : int32
        negative to fill interior, positive for thickness of the boundary
    debug : bool
        If True, output an uint8 mask image with 0 being negative and 255 being positive. Otherwise, output a float32 mask image with 0.0 being negative and 1.0 being positive.
    
    Returns
    -------
    numpy.ndarray
        a 2D array of resolution `out_imgres` representing the mask
    '''
    int_contours = [x.astype(_np.int32) for x in contours]
    if debug:
        mask = _np.zeros((out_imgres[1], out_imgres[0]), dtype=_np.uint8)
        _cv.drawContours(mask, int_contours, -1, 255, thickness)
    else:
        mask = _np.zeros((out_imgres[1], out_imgres[0]), dtype=_np.float32)
        _cv.drawContours(mask, int_contours, -1, 1.0, thickness)
    return mask


def polygon2mask(polygon, padding=0):
    '''Converts the interior of a polygon into an uint8 mask image with padding.

    Parameters
    ----------
    polygon : numpy.array
        list of 2D integer points (x,y)
    padding : int
        number of pixels for padding at all sides

    Returns
    -------
    img : numpy.array of shape (height, width)
        an uint8 2D image with 0 being zero and 255 being one representing the interior of the polygon, plus padding
    offset : numpy.array(shape=(2,))
        `(offset_x, offset_y)`. Each polygon's interior pixel is located at `img[offset_y+y,m offset_x+x]` and with value 255
    '''
    # compliance
    polygon = polygon.astype(_np.int32)

    # estimate boundaries
    tl = polygon.min(axis=0)
    br = polygon.max(axis=0)
    offset = tl - padding
    width, height = br + (padding+1) - offset
    polygon -= offset

    # draw polygon
    img = _np.zeros((height, width), dtype=_np.uint8)
    _cv.fillPoly(img, [polygon], 255)

    return img, offset


def morph_open(polygon, ksize=3):
    '''Applies a morphological opening operation on the interior of a polygon to form a more human-like polygon.

    Parameters
    ----------
    polygon : numpy.array
        list of 2D integer points (x,y)
    ksize : int
        size of morphological square kernel

    Returns
    -------
    polygons : list of numpy arrays
        list of output polygons, because morphological opening can split a thin polygon into a few parts
    '''
    # get the mask
    img, offset = polygon2mask(polygon, (ksize+1)//2)

    # morphological opening
    sem = _cv.getStructuringElement(_cv.MORPH_RECT, (ksize, ksize))
    img2 = _cv.morphologyEx(img, _cv.MORPH_OPEN, sem)

    contours, _ = _cv.findContours(img2, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
    #return img, img2, offset, contours, hier

    contours = [x.squeeze()+offset for x in contours]
    return contours

