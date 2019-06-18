# Data loader module

import numpy as np
import cv2
import os
import re

from .image_utils import readPFM, writePFM, sceneflowconstruct, preprocess
    
def datagen(folder_number):
    xtrain = []
    ytrain = []

    # Path of the datasets on the disk 
    img_dir = "/media/ravi/Dataset/data/frames_finalpass/TEST/A"
    of_dir  = "/media/ravi/Dataset/data/optical_flow/TEST/A"
    d_dir   = "/media/ravi/Dataset/data/disparity/TEST/A"

    # Camera parameters
    focal_length = 1050.0
    baseline = 1.0

    # To read pair of images according to sequence
    # In case the folder_number is a list.
    # The loop will go through all the folders to 
    # to create a batch
    # [folder_number] sqaure brackets make it a list and hence 
    # an interable
    for y in [folder_number]:
        
        # The following is for entering the 
        # respective folder number 
        if y<10:
            folder_path = "000%d" % (y)  
        elif y>=10 and y<100:
            folder_path = "00%d" % (y)
        elif y>=100 and y<1000:
            folder_path = "0%d" % (y)          
        else:
            folder_path = "%d" % (y)

        # Each folder has 15 images
        # Consecutive pairs from each folder is stacked together
        # Data is taken each from raw_imgages, optical_flow and disparity
        # im_path - image path, of_path - optical_flow and    
        for x in range(6, 14):
            if x < 9:
                im_path1 = "000%d.png" % (x)
                im_path2 = "000%d.png" % (x+1)
                of_path = "OpticalFlowIntoFuture_000%d_L.pfm"  % (x+1)
                disp_path1 = "000%d.pfm"% (x)
                disp_path2 = "000%d.pfm"% (x+1)           

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

            imgL1 = cv2.imread(os.path.join(img_dir, folder_path,"left", im_path1))
            imgR1 = cv2.imread(os.path.join(img_dir, folder_path,"right", im_path1)) # stereo pair at t
            imgL2 = cv2.imread(os.path.join(img_dir, folder_path,"left", im_path2))
            imgR2 = cv2.imread(os.path.join(img_dir, folder_path,"right", im_path2)) # stereo pair at t+1
                
            input_data = np.concatenate((imgL1, imgR1, imgL2, imgR2), axis = 2)

            # downsample
            input_data = preprocess(input_data)
            xtrain.append(input_data)
            
            of = readPFM(os.path.join(of_dir, folder_path, "into_future", "left", of_path))
            disp1 = readPFM(os.path.join(d_dir, folder_path, "left", disp_path1))
            disp2 = readPFM(os.path.join(d_dir, folder_path, "left", disp_path2))
            
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



