#!/usr/bin/python3

import argparse
from imghdr import what
from mt import cv, np

def get_image(imm):
    if imm.pixel_format in ['gray', 'bgr']:
        return imm.image
    
    if imm.pixel_format == 'rgb':
        return np.ascontiguousarray(np.flip(imm.image, axis=-1))
    
    if imm.pixel_format == 'rgba':
        h, w = imm.image.shape[:2]
        image = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        image[h:h*2,:w,0] = imm.image[:,:,2]
        image[h:h*2,:w,1] = imm.image[:,:,1]
        image[h:h*2,:w,2] = imm.image[:,:,0]

        image[:h,w:w*2,0] = imm.image[:,:,3]

        image[:h,:w,0] = np.round(imm.image[:,:,2].astype(float)*imm.image[:,:,3].astype(float)/255).astype(np.uint8)
        image[:h,:w,1] = np.round(imm.image[:,:,1].astype(float)*imm.image[:,:,3].astype(float)/255).astype(np.uint8)
        image[:h,:w,2] = np.round(imm.image[:,:,0].astype(float)*imm.image[:,:,3].astype(float)/255).astype(np.uint8)
        return image

    if imm.pixel_format == 'bgra':
        h, w = imm.image.shape[:2]
        image = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        image[h:h*2,:w,0] = imm.image[:,:,0]
        image[h:h*2,:w,1] = imm.image[:,:,1]
        image[h:h*2,:w,2] = imm.image[:,:,2]

        image[:h,w:w*2,0] = imm.image[:,:,3]

        image[:h,:w,0] = np.round(imm.image[:,:,0].astype(float)*imm.image[:,:,3].astype(float)/255).astype(np.uint8)
        image[:h,:w,1] = np.round(imm.image[:,:,1].astype(float)*imm.image[:,:,3].astype(float)/255).astype(np.uint8)
        image[:h,:w,2] = np.round(imm.image[:,:,2].astype(float)*imm.image[:,:,3].astype(float)/255).astype(np.uint8)
        return image

    raise ValueError("Imm with pixel format '{}' is not supported.".format(imm.pixel_format))

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
        view(get_image(imm), max_width=args.max_width)
    else:
        if args.imm_file.lower().endswith('.jp2'):
            image_type = 'JPEG2000'
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
