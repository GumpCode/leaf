import os
import sys
import random
import shutil
import copy
from PIL import Image

def main(argv):
    classList = os.listdir(argv[1])
    classNum = len(classList)
    trainList = []
    valList = []

    # make train&val dir
    os.mkdir(argv[2])
    os.mkdir(argv[3])

    #rename option
    for item in classList:
        count = 0
        imageDir = argv[1] + '/' + item
        imageList = os.listdir(imageDir)
        # shuffle the image list
        random.shuffle(imageList)
        # divide data into training and valuating
        for im in imageList:
            src = imageDir + '/' + im
            #image_name = item + '_' + im
            image_name = item + '_' + str(count) + '.jpg'
            # training data
            trainNum = int(float(argv[6]) * len(imageList))
            if count < trainNum:
                dst = argv[2] + '/' + image_name
                shutil.copy(src, dst)
                train = [image_name, item]
                trainList.append(copy.copy(train))
                count += 1

            # valuating data
            else:
                dst = argv[3] + '/' + image_name
                shutil.copy(src, dst)
                val = [image_name, item]
                valList.append(copy.copy(val))
                count += 1

    # make label txt
    with open(argv[4], 'w') as file:
        for item in trainList:
            file.write(item[0] + ' ' + item[1] + '\n')
    with open(argv[5], 'w') as file:
        for item in valList:
            file.write(item[0] + ' ' + item[1] + '\n')

if __name__ == '__main__':
    main(sys.argv)
