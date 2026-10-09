"""Extra functions dealing with polygons via OpenCV.

A polygon is defined as a list of 2D points `(x, y)`, not necessarily in integers, stored as a
numpy array of shape `(N, 2)`. We also define ndpoly (Nan delimited polygon) as a single array of
shape `(M, 2)` containing several polygons separated by rows of NaN points.

Examples
--------
>>> import numpy as np
>>> from mt.opencv.polygon import polygons2ndpoly, ndpoly2polygons
>>> tri = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
>>> sq = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=np.float32)
>>> ndpoly = polygons2ndpoly([tri, sq])
>>> ndpoly.shape
(8, 2)
>>> [p.shape for p in ndpoly2polygons(ndpoly)]
[(3, 2), (4, 2)]
"""

import shapely

from . import cv2 as _cv

from mt import tp, np


__all__ = [
    "polygons2ndpoly",
    "ndpoly2polygons",
    "mask2ndpoly",
    "ndpoly2mask",
    "ndpoly2MultiPolygon",
    "MultiPolygon2ndpoly",
    "render_mask",
    "polygon2mask",
    "morph_open",
]


def polygons2ndpoly(polygons: tp.List[np.ndarray]) -> np.ndarray:
    """Converts a list of polygons into an ndpoly (nan delimited polygon).

    Parameters
    ----------
    polygons : list
        a list of numpy arrays, each of which is a list of 2D points of shape `(N, 2)`, not
        necessarily in integers

    Returns
    -------
    numpy.ndarray
        a single numpy array of shape `(M, 2)` representing the ndpoly, where consecutive polygons
        are separated by a `[nan, nan]` row. If the list is empty, an empty `(0, 2)` float32 array
        is returned.

    See Also
    --------
    ndpoly2polygons : the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import polygons2ndpoly
    >>> tri = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    >>> polygons2ndpoly([tri, tri + 5])
    array([[ 0.,  0.],
           [ 1.,  0.],
           [ 1.,  1.],
           [nan, nan],
           [ 5.,  5.],
           [ 6.,  5.],
           [ 6.,  6.]])
    """
    ndpoly = []
    for poly in polygons:
        if len(ndpoly) > 0:
            ndpoly.append(np.array([[np.nan, np.nan]]))
        ndpoly.append(poly)
    if len(ndpoly) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(ndpoly)


def ndpoly2polygons(ndpoly: np.ndarray) -> tp.List[np.ndarray]:
    """Converts an ndpoly (nan delimited polygon) into a list of polygons.

    Parameters
    ----------
    ndpoly : numpy.ndarray
        a single numpy array of shape `(M, 2)` representing the ndpoly

    Returns
    -------
    list
        a list of numpy arrays, each of which is a list of 2D points, not necessarily in integers.
        Empty parts are dropped.

    See Also
    --------
    polygons2ndpoly : the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import ndpoly2polygons
    >>> nan = np.nan
    >>> ndpoly = np.array([[0, 0], [1, 0], [1, 1], [nan, nan], [5, 5], [6, 5], [6, 6]])
    >>> [p.tolist() for p in ndpoly2polygons(ndpoly)]
    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0]]]
    """
    if len(ndpoly) == 0:
        return []
    isnan = np.isnan(ndpoly).any(axis=1)
    split_indices = np.where(isnan)[0]
    polygons = []
    start_idx = 0
    for idx in split_indices:
        if idx > start_idx:
            polygons.append(ndpoly[start_idx:idx])
        start_idx = idx + 1
    if start_idx < len(ndpoly):
        polygons.append(ndpoly[start_idx:])
    return polygons


def mask2ndpoly(mask: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Converts a binary mask into an ndpoly (nan delimited polygon).

    The external contour of every connected component of the mask is extracted with
    :func:`cv2.findContours`, so holes are ignored.

    Parameters
    ----------
    mask : numpy.ndarray
        a 2D binary mask array of shape `(height, width)`. It is converted to uint8, so any non-zero
        value of a boolean or integer mask is foreground.
    epsilon : float, optional
        unused at the moment. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        a single float32 numpy array representing the ndpoly, with points in `(x, y)` order

    See Also
    --------
    ndpoly2mask : the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import mask2ndpoly
    >>> mask = np.zeros((5, 6), dtype=np.uint8)
    >>> mask[1:4, 1:5] = 1
    >>> mask2ndpoly(mask)
    array([[1., 1.],
           [1., 3.],
           [4., 3.],
           [4., 1.]], dtype=float32)
    """
    mask = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = _cv.findContours(mask, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.squeeze().astype(np.float32)
        polygons.append(contour)
    return polygons2ndpoly(polygons)


def render_mask(contours, out_imgres, thickness=-1, debug=False):
    """Renders a mask array from a list of contours.

    Parameters
    ----------
    contours : list
        a list of numpy arrays, each of which is a list of 2D points `(x, y)`, not necessarily in
        integers. The points are truncated to integers before drawing.
    out_imgres : list
        the `[width, height]` image resolution of the output mask.
    thickness : int, optional
        negative to fill interior, positive for thickness of the boundary. Default is -1.
    debug : bool, optional
        If True, output an uint8 mask image with 0 being negative and 255 being positive. Otherwise,
        output a float32 mask image with 0.0 being negative and 1.0 being positive. Default is
        False.

    Returns
    -------
    numpy.ndarray
        a 2D array of shape `(out_imgres[1], out_imgres[0])` representing the mask

    See Also
    --------
    ndpoly2mask : the same for an ndpoly

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import render_mask
    >>> render_mask([np.array([[0, 0], [2, 0], [2, 2]])], [4, 3])
    array([[1., 1., 1., 0.],
           [0., 1., 1., 0.],
           [0., 0., 1., 0.]], dtype=float32)
    """
    int_contours = [x.astype(np.int32) for x in contours]
    if debug:
        mask = np.zeros((out_imgres[1], out_imgres[0]), dtype=np.uint8)
        _cv.drawContours(mask, int_contours, -1, 255, thickness)
    else:
        mask = np.zeros((out_imgres[1], out_imgres[0]), dtype=np.float32)
        _cv.drawContours(mask, int_contours, -1, 1.0, thickness)
    return mask


def ndpoly2mask(
    ndpoly: np.ndarray,
    out_imgres: tp.List[int],
    thickness: int = -1,
    debug: bool = False,
) -> np.ndarray:
    """Renders a mask array from an ndpoly (nan delimited polygon).

    Parameters
    ----------
    ndpoly : numpy.ndarray
        a single numpy array representing the ndpoly
    out_imgres : list
        the `[width, height]` image resolution of the output mask.
    thickness : int, optional
        negative to fill interior, positive for thickness of the boundary. Default is -1.
    debug : bool, optional
        If True, output an uint8 mask image with 0 being negative and 255 being positive. Otherwise,
        output a float32 mask image with 0.0 being negative and 1.0 being positive. Default is
        False.

    Returns
    -------
    numpy.ndarray
        a 2D array of shape `(out_imgres[1], out_imgres[0])` representing the mask

    See Also
    --------
    mask2ndpoly : the inverse operation
    render_mask : the same for a list of polygons

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import ndpoly2mask
    >>> ndpoly = np.array([[1, 1], [4, 1], [4, 3], [1, 3]])
    >>> ndpoly2mask(ndpoly, [6, 5], debug=True)
    array([[  0,   0,   0,   0,   0,   0],
           [  0, 255, 255, 255, 255,   0],
           [  0, 255, 255, 255, 255,   0],
           [  0, 255, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    contours = ndpoly2polygons(ndpoly)
    return render_mask(contours, out_imgres, thickness, debug)


def ndpoly2MultiPolygon(ndpoly: np.ndarray) -> shapely.MultiPolygon:
    """Converts an ndpoly (nan delimited polygon) into a Shapely MultiPolygon.

    Parts with fewer than 3 points are skipped.

    Parameters
    ----------
    ndpoly : numpy.ndarray
        a single numpy array representing the ndpoly

    Returns
    -------
    shapely.MultiPolygon
        a Shapely MultiPolygon object, empty if there is no valid part

    See Also
    --------
    MultiPolygon2ndpoly : the inverse operation

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import ndpoly2MultiPolygon
    >>> nan = np.nan
    >>> ndpoly = np.array([[0, 0], [1, 0], [1, 1], [nan, nan], [5, 5], [6, 5], [6, 6]])
    >>> print(ndpoly2MultiPolygon(ndpoly))
    MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((5 5, 6 5, 6 6, 5 5)))
    """
    polygons = ndpoly2polygons(ndpoly)
    shapely_polygons = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        shapely_polygons.append(shapely.Polygon(poly))
    if len(shapely_polygons) == 0:
        return shapely.MultiPolygon()
    return shapely.MultiPolygon(shapely_polygons)


def MultiPolygon2ndpoly(multipolygon: shapely.MultiPolygon) -> np.ndarray:
    """Converts a Shapely MultiPolygon into an ndpoly (nan delimited polygon).

    Only the exterior ring of each polygon is kept (holes are dropped) and the duplicated closing
    point of the ring is removed. A single Shapely Polygon is also accepted, and non-polygon
    geometries are skipped.

    Parameters
    ----------
    multipolygon : shapely.MultiPolygon
        a Shapely MultiPolygon object

    Returns
    -------
    numpy.ndarray
        a single float32 numpy array representing the ndpoly

    See Also
    --------
    ndpoly2MultiPolygon : the inverse operation
    """
    if isinstance(multipolygon, shapely.Polygon):
        multipolygon = shapely.MultiPolygon([multipolygon])

    polygons = []
    for poly in multipolygon.geoms:
        if isinstance(poly, shapely.Polygon) is False:
            continue
        exterior_coords = np.array(poly.exterior.coords, dtype=np.float32)
        exterior_coords = exterior_coords[:-1]  # remove duplicated last point
        polygons.append(exterior_coords)
    return polygons2ndpoly(polygons)


def polygon2mask(polygon, padding=0):
    """Converts the interior of a polygon into an uint8 mask image with padding.

    The mask is the tight bounding box of the polygon plus `padding` pixels at all sides.

    Parameters
    ----------
    polygon : numpy.ndarray
        list of 2D integer points `(x, y)`, of shape `(N, 2)`. It is converted to int32.
    padding : int, optional
        number of pixels for padding at all sides. Default is 0.

    Returns
    -------
    img : numpy.ndarray of shape (height, width)
        an uint8 2D image with 0 being zero and 255 being one representing the interior of the
        polygon, plus padding
    offset : numpy.ndarray of shape (2,)
        `(offset_x, offset_y)`. Each polygon's interior pixel is located at
        `img[y - offset_y, x - offset_x]` and with value 255, that is, the offset is the position
        in polygon coordinates of the top-left pixel of the mask.

    Examples
    --------
    >>> import numpy as np
    >>> from mt.opencv.polygon import polygon2mask
    >>> img, offset = polygon2mask(np.array([[2, 2], [5, 2], [5, 4], [2, 4]]), padding=1)
    >>> img
    array([[  0,   0,   0,   0,   0,   0],
           [  0, 255, 255, 255, 255,   0],
           [  0, 255, 255, 255, 255,   0],
           [  0, 255, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0,   0]], dtype=uint8)
    >>> offset.tolist()
    [1, 1]
    """
    # compliance
    polygon = polygon.astype(np.int32)

    # estimate boundaries
    tl = polygon.min(axis=0)
    br = polygon.max(axis=0)
    offset = tl - padding
    width, height = br + (padding + 1) - offset
    polygon -= offset

    # draw polygon
    img = np.zeros((height, width), dtype=np.uint8)
    _cv.fillPoly(img, [polygon], 255)

    return img, offset


def morph_open(polygon, ksize=3):
    """Applies a morphological opening operation on the interior of a polygon to form a more
    human-like polygon.

    Parameters
    ----------
    polygon : numpy.ndarray
        list of 2D integer points `(x, y)`
    ksize : int, optional
        size of morphological square kernel. Default is 3.

    Returns
    -------
    polygons : list of numpy.ndarray
        list of output polygons, in the coordinates of the input polygon, because morphological
        opening can split a thin polygon into a few parts
    """
    # get the mask
    img, offset = polygon2mask(polygon, (ksize + 1) // 2)

    # morphological opening
    sem = _cv.getStructuringElement(_cv.MORPH_RECT, (ksize, ksize))
    img2 = _cv.morphologyEx(img, _cv.MORPH_OPEN, sem)

    contours, _ = _cv.findContours(img2, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
    # return img, img2, offset, contours, hier

    contours = [x.squeeze() + offset for x in contours]
    return contours
