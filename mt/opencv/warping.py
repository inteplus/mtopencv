'''Affine-warping and cropping an image.'''


import mt.geo.affine2d as ga
from . import cv2 as cv


__all__ = ['warp_image', 'crop_image']


def warp_image(out_image, in_image, warp_tfm, border_mode='constant'):
    '''Takes a crop window from input image and warp/resize it to output image.
    
    Parameters
    ----------
    out_image : numpy.ndarray
        output image to be warped and resized to
    in_image : numpy.ndarray
        input image from which the warping takes place
    warp_tfm : mt.geo.affine2d.Aff2d
        2D transformation mapping pixel locations in the input image to the `[0,1]^2` square
    border_mode : {'constant', 'replicate'}
        border filling mode. 'constant' means filling zero constant. 'replicate' means replicating last pixels in each dimension. 
    '''    
    inv_tfm = ~(ga.scale2d(out_image.shape[1], out_image.shape[0])*warp_tfm)
    borderMode = cv.BORDER_CONSTANT if border_mode == 'constant' else cv.BORDER_REPLICATE
    cv.warpAffine(
        in_image,
        inv_tfm.matrix[:2, :],
        dst=out_image,
        dsize=(out_image.shape[1], out_image.shape[0]),
        flags=cv.WARP_INVERSE_MAP | cv.INTER_NEAREST,
        borderMode=borderMode) # we fill zeros here


def crop_image(out_image, in_image, crop_rect, border_mode='constant'):
    '''Takes a crop window from input image and warp/resize it to output image.
    
    Parameters
    ----------
    out_image : numpy.ndarray
        output image to be cropped and resized to
    in_image : numpy.ndarray
        input image from which the crop takes place
    crop_rect : mt.geo.rect.Rect
        crop window
    border_mode : {'constant', 'replicate'}
        border filling mode. 'constant' means filling zero constant. 'replicate' means replicating last pixels in each dimension. 
    '''    
    crop_tfm = ga.crop2d((crop_rect.min_x, crop_rect.min_y), (crop_rect.max_x, crop_rect.max_y))
    return warp_image(out_image, in_image, crop_tfm, border_mode=border_mode)
