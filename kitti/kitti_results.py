import sys
sys.path.append("/home/traviku2/conditional_gan")

import os
import datetime
import numpy as np
import cv2

import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from viz.colorcode import flow_to_color 
from image_utils import kitti_flow_write, kitti_disp_write

now = datetime.datetime.now()

def eval_results(gen_sceneflow, of, d1, d2):

    ###### Gets the predicted and ground truth scene flow with stereo images

    optical_flow = gen_sceneflow[:, :, 0:3]  #
    disparity1   = gen_sceneflow[:, :, 3]    # Predictions
    disparity2   = gen_sceneflow[:, :, 4]    #

    u = optical_flow[:,:,0]
    v = optical_flow[:,:,1]

    of_outpath = os.path.join('/home/traviku2/kitti/results/flow/', of)
    d1_outpath = os.path.join('/home/traviku2/kitti/results/disp_0/', d1)
    d2_outpath = os.path.join('/home/traviku2/kitti/results/disp_1/', d2)

    kitti_flow_write(u,v, of_outpath, valid=None)
    kitti_disp_write(disparity1, d1_outpath)
    kitti_disp_write(disparity2, d2_outpath)


   
