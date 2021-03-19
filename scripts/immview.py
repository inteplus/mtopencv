#!/usr/bin/python3

import argparse
from imghdr import what
from mt import cv, np

def main(args):
    if args.imm_file.endswith('.imm'):
        imm = cv.immload(args.imm_file)
        print("Image path: {}".format(args.imm_file))
        print("Pixel format: {}".format(imm.pixel_format))
        print("Meta:")
        print(imm.meta)
        print("Press any key to exit.")
        cv.namedWindow('image')
        cv.imshow('image', imm.image)
        cv.waitKey(0)
    else:
        image_type = what(args.imm_file)
        if image_type is None:
            print("Not an image file: {}".format(args.imm_file))
        else:
            print("Image path: {}".format(args.imm_file))
            print("File type: {}".format(image_type))
            image = cv.imread(args.imm_file)
            print("Press any key to exit.")
            cv.namedWindow('image')
            cv.imshow('image', image)
            cv.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool to view an image with metadata (IMM) file.")
    parser.add_argument('imm_file', type=str,
                        help="The file to view.")
    args = parser.parse_args()    
    main(args)
