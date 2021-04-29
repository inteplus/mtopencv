'''Converting an RGB image into ANSI format.'''

from colors import color

def to_ansi(img, max_imgres=None):
    '''Converts an RGB image into ANSI format for displaying on a terminal.

    Given an image, the function first determines the resolution to which the image
    will be resized. After resizing to the new resolution, each pixel is converted
    into a white space with a 24-bit color, using ANSI encoding.

    Parameters
    ----------
    img : numpy.ndarray
        an RGB uint8 image
    max_imgres : list, optional
        pair of [width, height] defining the maximum resolution. If not specified,
        we estimate from the current terminal.

    Returns
    -------
    str
        a multi-line string that can be printed to a terminal
    '''
    pass
