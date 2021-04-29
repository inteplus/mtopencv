'''Converting an RGB image into ANSI format.'''

from colors import color
import cv2 as cv

from mt.base.terminal import stty_imgres


__all__ = ['to_ansi']


def get_pixel(col):
    '''Converts a pixel into an ANSI letter.'''
    return color(' ', bg=f'rgb({int(col[0])}, {int(col[1])}, {int(col[2])})')


def to_ansi(img, imgres=None):
    '''Converts an RGB image into ANSI format for displaying on a terminal.

    Given an image, the function first determines the resolution to which the image
    will be resized. After resizing to the new resolution, each pixel is converted
    into a white space with a 24-bit color, using ANSI encoding. Note that each white
    space is viewed as a 4x8 block, encouraging the width to be twice larger than
    expected.

    Parameters
    ----------
    img : numpy.ndarray
        an RGB uint8 image
    imgres : list, optional
        pair of [width, height] defining the target resolution. If not specified,
        we estimate from the current terminal.

    Returns
    -------
    str
        a multi-line string that can be printed to a terminal
    '''

    # determine the resolution
    if imgres is None:
        img_imgres = [img.shape[1], img.shape[0]]

        imgres = stty_imgres()

        # reduce a bit if we can
        for i in range(2):
            if imgres[i] > 10:
                imgres[i] -= 2

        # width:height aspect ratio
        ratio = img_imgres[0]*2/img_imgres[1]

        # adjust width or height accordingly
        width = int(imgres[1]*ratio)
        if imgres[0] > width:
            imgres[0] = width

        height = int(imgres[0]/ratio)
        if imgres[1] > height:
            imgres[1] = height

    # resize the image
    img = cv.resize(img, (imgres[0], imgres[1]))

    # encode
    output = ''
    for line in img:
        for pixel in line:
            output += get_pixel(pixel)
        output += '\n'

    return output
