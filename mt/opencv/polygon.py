"""Extra functions dealing with polygons via OpenCV.

A polygon is defined as a list of 2D points, not necessarily in integers. We also define ndpoly
(Nan delimited polygon) as a polygon that may contain NaN points to separate different parts.
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
        a list of numpy arrays, each of which is a list of 2D points, not necessarily in integers

    Returns
    -------
    numpy.ndarray
        a single numpy array representing the ndpoly
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
        a single numpy array representing the ndpoly

    Returns
    -------
    list
        a list of numpy arrays, each of which is a list of 2D points, not necessarily in integers
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

    Parameters
    ----------
    mask : numpy.ndarray
        a 2D binary mask array
    epsilon : float
        the approximation accuracy parameter for polygonal approximation

    Returns
    -------
    numpy.ndarray
        a single numpy array representing the ndpoly
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
        a list of numpy arrays, each of which is a list of 2D points, not necessarily in integers
    out_imgres : list
        the [width, height] image resolution of the output mask.
    thickness : int32
        negative to fill interior, positive for thickness of the boundary
    debug : bool
        If True, output an uint8 mask image with 0 being negative and 255 being positive. Otherwise,
        output a float32 mask image with 0.0 being negative and 1.0 being positive.

    Returns
    -------
    numpy.ndarray
        a 2D array of resolution `out_imgres` representing the mask
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
        the [width, height] image resolution of the output mask.
    thickness : int32
        negative to fill interior, positive for thickness of the boundary
    debug : bool
        If True, output an uint8 mask image with 0 being negative and 255 being positive. Otherwise,
        output a float32 mask image with 0.0 being negative and 1.0 being positive.

    Returns
    -------
    numpy.ndarray
        a 2D array of resolution `out_imgres` representing the mask
    """
    contours = ndpoly2polygons(ndpoly)
    return render_mask(contours, out_imgres, thickness, debug)


def ndpoly2MultiPolygon(ndpoly: np.ndarray) -> shapely.MultiPolygon:
    """Converts an ndpoly (nan delimited polygon) into a Shapely MultiPolygon.

    Parameters
    ----------
    ndpoly : numpy.ndarray
        a single numpy array representing the ndpoly

    Returns
    -------
    shapely.MultiPolygon
        a Shapely MultiPolygon object
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

    Parameters
    ----------
    multipolygon : shapely.MultiPolygon
        a Shapely MultiPolygon object

    Returns
    -------
    numpy.ndarray
        a single numpy array representing the ndpoly
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
    """Applies a morphological opening operation on the interior of a polygon to form a more human-like polygon.

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
