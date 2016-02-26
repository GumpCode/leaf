#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "label_file",
        help="label txt for test data"
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
                "leaf_full_iter_50000.caffemodel"),
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

#if args.gpu:
    caffe.set_mode_gpu()
    print("GPU mode")
#else:
#caffe.set_mode_cpu()
#print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Read label file
    with open(args.label_file, 'r') as file:
        label = []
        image_list = []
        for line in file.readlines():
            temp = line.strip('\n').split(' ')
            image_list.append(args.input_file + '/' + temp[0])
            label.append(temp[1])

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in image_list]
    else:
        print("Loading file: %s" % args.input_file)
        inputs = [caffe.io.load_image(args.input_file)]

    print("Classifying %d inputs." % len(inputs))

    iteration = 10000
    start = time.time()
    correct = 0.0
    wrong = 0.0
    for num in range(iteration):
        # Classify.
        start = time.time()
        print len(inputs[1]), np.shape(inputs[1])    
        predictions = classifier.predict(inputs, not args.center_only)
        sys.stdout.write("Iteration " + str(num) + " in %.2f s." % (time.time() - start))
        print(' ')
        i = 0
        count = 0
        for output in predictions:
            answer = '3'
            if (float(output[0]) > float(output[1])):
                answer = '0'
            else:
                answer = '1'
            
            if (answer == label[i]):
                correct = correct + 1
            else:
                wrong = wrong + 1
                print output, image_list[i], label[i]
                count = count + 1
            i = i + 1
        print count

        sys.stdout.write("Accuracy rate is %.4f" % (correct/(correct+wrong)))
        print(' ')



if __name__ == '__main__':
    main(sys.argv)
