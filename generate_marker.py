# coding: utf-8

from cv2 import aruco as ar
from skimage import io
import argparse


parser = argparse.ArgumentParser(description='Generates aruco marker.')
parser.add_argument('id', type=int, metavar='i', help='id of marker',
                    choices=list(range(250)))
parser.add_argument('--size', type=int, metavar='s',
                    help='size of marker in pixel',
                    default=200, choices=list(range(1000)))
args = parser.parse_args()

print("Generating aruco marker with id {:d} and size {:d}."
      .format(args.id, args.size))

dict = ar.getPredefinedDictionary(ar.DICT_4X4_250)
marker = ar.drawMarker(dict, args.id, args.size)
io.imsave(str(args.id)+'.png', marker)
