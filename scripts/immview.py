#!/usr/bin/python3

import argparse
from imghdr import what
from mt import cv, np

def view(image, max_width=640):
    if max_width < image.shape[1]:
        height = image.shape[0]*max_width//image.shape[1]
        image = cv.resize(image, dsize=(max_width, height))
    cv.namedWindow('image')
    print("Press any key to exit.")
    cv.imshow('image', image)
    cv.waitKey(0)

def main(args):
    if args.imm_file.endswith('.imm'):
        imm = cv.immload(args.imm_file)
        print("Image path: {}".format(args.imm_file))
        print("Pixel format: {}".format(imm.pixel_format))
        print("Resolution: {}x{}".format(imm.image.shape[1], imm.image.shape[0]))
        print("Meta:")
        print(imm.meta)
        view(imm.image, max_width=args.max_width)
    else:
        image_type = what(args.imm_file)
        if image_type is None:
            print("Not an image file: {}".format(args.imm_file))
        else:
            print("Image path: {}".format(args.imm_file))
            print("File type: {}".format(image_type))
            image = cv.imread(args.imm_file)
            print("Resolution: {}x{}".format(image.shape[1], image.shape[0]))
            view(image, max_width=args.max_width)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to view an image with metadata (IMM) file.")
    parser.add_argument('--max_width', type=int, default=640,
                        help="The maximum width to view. Default is 640.")
    parser.add_argument('imm_file', type=str,
                        help="The file to view.")
    args = parser.parse_args()    
    main(args)
