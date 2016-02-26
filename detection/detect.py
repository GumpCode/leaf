#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import copy
import shutil
import os
import sys
import argparse
import glob
import time
import cv2

import caffe
        
threshold = 0.95
window_threshold = 150
areaThreshold = 0.8
initial_window_scale = 32
scale_step = 10
height_step = 8
width_step = 8
save_Path='/home/ganlinhao/DeepLearning/caffe/examples/leaf/Image'

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )

    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "leaf_full.prototxt"),
        help="Model definition file."
    )

    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "leaf_full_iter_40000.caffemodel"),
        help="Trained model weights file."
    )

    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )

    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )

    parser.add_argument(
        "--images_dim",
        default='32,32',
        help="Canonical 'height,width' dimensions of input images."
    )

    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'leaf_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )

    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )

    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )

    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )

    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )

    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    caffe.set_device(1)
    caffe.set_mode_gpu()
    print("GPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs = glob.glob(args.input_file + '/*')
    else:
        print("Loading file: %s" % args.input_file)
        inputs = glob.glob(args.input_file)

    print("Loaded %d inputs." % len(inputs))

    if os.path.exists(save_Path):
        shutil.rmtree(save_Path)
    os.mkdir(save_Path)
    start = time.time()
    im_num = 0
    for image in inputs:
        im = caffe.io.load_image(image)
        window_scale = initial_window_scale
        detections = []
        num = 0
        while(window_scale < window_threshold):
            # ymin xmin ymax xmax
            window=[0, 0, window_scale, window_scale]
            flag = True
            while(flag == True):
                # Crop image
                crop = im[window[0]:window[2], window[1]:window[3]]
                imgs = caffe.io.resize_image(crop, (32, 32))
                imgs_list = []
                imgs_list.append(imgs)
                # predict
                prediction = classifier.predict(imgs_list, not args.center_only)
                
                # chose the rectangle window
                if prediction[0][1] > threshold:
                    isAdd = True
                    if len(detections) == 0:
                        detections.append(copy.copy(window))
                    else:
                        for win in detections:
                            isOverlap, window = merge_rectangle(win, window)
                            if isOverlap: 
                                detections.remove(win)
                                result = copy.copy(window)
                                isAdd = False
                        if isAdd:
                            detections.append(copy.copy(window))
                        else:
                            detections.append(copy.copy(result))
                    num += 1
                window, flag = move_window(window, np.shape(im), window_scale)
            window_scale = window_scale + scale_step

        # recheck the overlap rectangle
        remove = []
        for i in range(0, len(detections)):
            for j in range(i+1, len(detections)):
                isRepeated, newWin = merge_rectangle(detections[i], detections[j])
                if isRepeated:
                    remove.append(copy.copy(detections[i]))
        for win in remove:
            if win in detections:
                detections.remove(win)
        
        # draw the rectangles
        draw_rectangle(detections, image)
        print detections
        print ('the No.' + str(im_num) + ' image is completed:')
        print ('    ' + str(len(detections)) + ' windows are detected!')
        im_num += 1



def move_window(window, shape, window_scale):
    flag = True
    if (window[3] + (width_step)) < shape[1]:
#window[1] = window[1] + (window_scale/2)
        window[1] = window[1] + width_step
#window[3] = window[3] + (window_scale/2)
        window[3] = window[3] + width_step
    else:
        window[1] = 0
        window[3] = window_scale
        if (window[2]+ (height_step)) < shape[0]:
            window[0] = window[0] + height_step
#window[0] = window[0] + (window_scale/2)
            window[2] = window[2] + height_step
#            window[2] = window[2] + (window_scale/2)
        else:
            flag = False

    return window, flag


def draw_rectangle(detectedList, image):
    # read the image
    im = cv2.imread(image)
    if not len(detectedList) == 0:
        # draw the rectangle
        for window in detectedList:
            cv2.rectangle(im, (window[1], window[2]), (window[3], window[0]), (0,0,255), 5)
    # save the image
    cv2.imwrite(save_Path + '/' + image.split('/')[len(image.split('/'))-1], im)


def merge_rectangle(window1, window2):
    isOverlap = False
    result = copy.copy(window2)
    # find the overlap area
    if not ((window1[1] > window2[3]) | (window2[1] > window1[3]) | (window1[0] > window2[2]) | (window2[0] > window1[2])):
        for i in range(4):
            if window1[i] > window2[i]:
                result[i] = window1[i]
            else:
                result[i] = window2[i]

        # compute the area of rectangles
        overlapArea = float((result[2] - result[0]) * (result[3] - result[1]))
        Area1 = float((window1[2] - window1[0]) * (window1[3] - window1[1]))
        Area2 = float((window2[2] - window2[0]) * (window2[3] - window2[1]))

        # select the window
        if ((overlapArea/Area1) > areaThreshold) | ((overlapArea/Area2) > areaThreshold):
            if ((window1[3]-window1[1]) * (window1[2]-window1[0]) > (window2[3]-window2[1]) * (window2[2]-window2[0])):
                for j in range(4):
                    result[j] = copy.copy(window1[j])
            else:
                for j in range(4):
                    result[j] = copy.copy(window2[j])
            isOverlap = True

    return isOverlap, result



if __name__ == '__main__':
    main(sys.argv)
