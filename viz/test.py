# Code to visualize Optical flow and Disparity 
# 
# Adapted from 
# Author: Tom Runia
# Date Created: 2018-08-03

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
import os
sys.path.append(os.getcwd())

#import keras.backend as K
import numpy as np

#import cv2
#mport matplotlib.pyplot as plt
#rom mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io


from viz.visualization import flow_to_color, vis_opticalflow
from sfgan.image_utils import readPFM
from sfgan.utils import datagen
#from viz.visualization import flow_to_color
from sfgan.image_utils import readPFM
#from sfgan.utils import datagen

'''
im, sf = datagen(1)

a = np.random.permutation(8)
batch_size = 4

for i in range(int(8/batch_size)):
    index = a[i*batch_size:(i+1)*batch_size]
    xtrain = im[index]
    ytrain = sf[index]
    print index, xtrain.shape, ytrain.shape
'''

a = readPFM('../scripts/d1.pfm')
b = readPFM('../scripts/of.pfm')

scipy.io.savemat('of.mat', mdict={'b':b})
scipy.io.savemat('d1.mat', mdict={'a':a})

plt.figure()
plt.imshow(a, cmap='gray')
vis_opticalflow(b)
plt.show()
'''


u = readPFM('../samples/dX.pfm')
v = readPFM('../samples/dY.pfm')
w = readPFM('../samples/dZ.pfm')

plt.show()

quiver3d(u, v, w)
