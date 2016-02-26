import cv2
import os
import sys

"
input: dim dim srcDir distDir
"

def main(argv):
    list = os.listdir(argv[3])

    if not os.path.exists(argv[4]):
        os.mkdir(argv[4])

    num = 0
    for items in list:
        im = cv2.imread(argv[3] + '/' + items)
        src = cv2.resize(im, (int(argv[1]), int(argv[2])))
        cv2.imwrite(argv[4] + '/' + items, src)
        num = num + 1

    print(str(num) + ' pictures were processed!')

if __name__ == '__main__':
    main(sys.argv)
