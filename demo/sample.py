import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from inout import preprocess, read

a = read('/media/ravi/Data/SCENEFLOW/prediction_gan/KITTI/optical_flow/left/OpticalFlowIntoFuture_0007_L.png')
b = read('/media/ravi/Data/SCENEFLOW/prediction_gan/KITTI/disparity/left/0006.png')
c = read('/media/ravi/Data/SCENEFLOW/prediction_gan/KITTI/disparity/left/0007.png')

print a.dtype
print b.dtype
print c.dtype

disp = np.dstack((b, c))

print disp.shape

def process(img):
    h, w, c = img.shape
    if h<385:
        img = cv2.copyMakeBorder(img, 384-h, 0, 0, 0, cv2.BORDER_CONSTANT)
        #img = np.pad(img, ((384-h, 0), (0,0), (0,0)), 'constant')
    print img.shape
    img = cv2.resize(img, (512,384))

    return img

#of   = process(a)
disp = process(disp)

#output = np.dstack((of, disp))

print disp.shape
