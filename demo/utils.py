# Data loader module

import numpy as np
import cv2
import os
import re

#from sfgan.image_utils import readPFM, writePFM, preprocess
from .inout import read, preprocess
    
def datagen(path):
    xtrain = []
    ytrain = []


    # Camera parameters
    focal_length = 1050.0
    baseline = 1.0

        # Each folder has 15 images
        # Consecutive pairs from each folder is stacked together
        # Data is taken each from raw_imgages, optical_flow and disparity
        # im_path - image path, of_path - optical_flow and    
    for x in range(6, 7):
        if x < 9:
            im_path1 = "000%d.png" % (x)
            im_path2 = "000%d.png" % (x+1)
            of_path = "OpticalFlowIntoFuture_000%d_L.png"  % (x+1)
            disp_path1 = "000%d.png"% (x)
            disp_path2 = "000%d.png"% (x+1)           

        elif x > 9:
            im_path1 = "00%d.png" % (x)
            im_path2 = "00%d.png" % (x+1)
            of_path = "OpticalFlowIntoFuture_00%d_L.pfm" % (x+1)
            disp_path1 = "00%d.pfm"%(x)
            disp_path2 = "00%d.pfm"%(x+1)  

        else:      # for x=9, gives frame 9 and 10
            im_path1 = "000%d.png" % (x)
            im_path2 = "00%d.png" % (x+1)
            of_path = "OpticalFlowIntoFuture_00%d_L.pfm" % (x+1)
            disp_path1 = "000%d.pfm"%(x)
            disp_path2 = "00%d.pfm"%(x+1)       

        imgL1 = read(os.path.join(path, "images/left", im_path1))
        imgR1 = read(os.path.join(path, "images/right",im_path1)) # stereo pair at t
        imgL2 = read(os.path.join(path, "images/left", im_path2))
        imgR2 = read(os.path.join(path, "images/right", im_path2)) # stereo pair at t+1
                
        input_data = np.concatenate((imgL1, imgR1, imgL2, imgR2), axis = 2)

        # downsample
        input_data = preprocess(input_data)
        xtrain.append(input_data)
        '''
        of    - optical flow between imgL1 and imgL2
        disp1 - disparity between imgL1 and imgR1
        disp2 - disparity between imgL2 and imgR2 
        '''
        of = read(os.path.join(path,"optical_flow/left", of_path))
        disp1 = read(os.path.join(path,"disparity/left", disp_path1)) 
        disp2 = read(os.path.join(path,"disparity/left", disp_path2))
        
        #target = sceneflowconstruct(of, disp1, disp2)
        #depth1 = (focal_length*baseline)/disp1
        #depth2 = (focal_length*baseline)/disp2 
            
        target = np.dstack((of, disp1, disp2))
        # downsample
        target = preprocess(target)
        ytrain.append(target)

    # xtrain is the stack of input images
    xtrain = np.array(xtrain)
    # ytrain is teh stack of sceneflow 
    ytrain = np.array(ytrain)
    #print y
    #print xtrain.shape, ytrain.shape
    return xtrain, ytrain



