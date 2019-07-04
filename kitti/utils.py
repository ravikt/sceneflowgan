# Data loader module

import numpy as np
import cv2
import os
import re

from image_utils import kitti_disp_read, kitti_flow_read, preprocess
    
def datagen(dataset_path, x):
    xtrain = []
    ytrain = []

    # Path of the datasets on the disk 
    img_dir = "images"
    of_dir  = "optical_flow"
    d_dir   = "disparity"
    # To read pair of images according to sequence
    # In case the folder_number is a list.
    # The loop will go through all the folders to 
    # to create a batch
    # [folder_number] sqaure brackets make it a list and hence 
    # an interable
    if x < 10:
        im_path1 = "00000%d_10.png" % (x)
        im_path2 = "00000%d_11.png" % (x)
        of_path = "00000%d_10.png"  % (x)
        disp_path1 = "00000%d_10.png"% (x)
        disp_path2 = "00000%d_10.png"% (x)           

    elif x > 9 and x <100:
        im_path1 = "0000%d_10.png" % (x)
        im_path2 = "0000%d_11.png" % (x)
        of_path = "0000%d_10.png" % (x)
        disp_path1 = "0000%d_10.png"%(x)
        disp_path2 = "0000%d_10.png"%(x)  

    else:      # for x=9, gives frame 9 and 10
        im_path1 = "000%d_10.png" % (x)
        im_path2 = "000%d_11.png" % (x)
        of_path = "000%d_10.png" % (x)
        disp_path1 = "000%d_10.png"%(x)
        disp_path2 = "000%d_10.png"%(x)       
        
    imgL1 = cv2.imread(os.path.join(dataset_path, img_dir,"left", im_path1))
    imgR1 = cv2.imread(os.path.join(dataset_path, img_dir,"right", im_path1)) # stereo pair at t
    imgL2 = cv2.imread(os.path.join(dataset_path, img_dir,"left", im_path2))
    imgR2 = cv2.imread(os.path.join(dataset_path, img_dir,"right", im_path2)) # stereo pair at t+1
        
    input_data = np.concatenate((imgL1, imgR1, imgL2, imgR2), axis = 2)

    # downsample
    input_data = preprocess(input_data)
    xtrain.append(input_data)
    
    of = kitti_flow_read(os.path.join(dataset_path, of_dir, of_path))
    disp1 = kitti_disp_read(os.path.join(dataset_path, d_dir,"disp_0", disp_path1)) # at t
    disp2 = kitti_disp_read(os.path.join(dataset_path, d_dir,"disp_1", disp_path2)) # at t+1
    
    target = np.dstack((of, disp1, disp2))
    # downsample
    target = preprocess(target)
    ytrain.append(target)

    # xtrain is the stack of input images
    xtrain = np.array(xtrain)
    # ytrain is the stack of sceneflow 
    ytrain = np.array(ytrain)
    #print y
    #print xtrain.shape, ytrain.shape
    return xtrain, ytrain
    #return xtrain, ytrain, of_path, disp_path1, disp_path2

def datagen_eval(dataset_path, x):
    """
    Data loader function for running predictions
    for test data
    """
    xtrain = []

    # Path of the datasets on the disk 
    # To read pair of images according to sequence
    # In case the folder_number is a list.
    # The loop will go through all the folders to 
    # to create a batch
    # [folder_number] sqaure brackets make it a list and hence 
    # an interable
    if x < 10:
        im_path1 = "00000%d_10.png" % (x)
        im_path2 = "00000%d_11.png" % (x)
        of_path = "00000%d_10.png"  % (x)
        disp_path = "00000%d_10.png"% (x)

    elif x > 9 and x <100:
        im_path1 = "0000%d_10.png" % (x)
        im_path2 = "0000%d_11.png" % (x)
        of_path = "0000%d_10.png"  % (x)
        disp_path = "0000%d_10.png"% (x) 

    else:      # for x=9, gives frame 9 and 10
        im_path1 = "000%d_10.png" % (x)
        im_path2 = "000%d_11.png" % (x)
        of_path = "000%d_10.png"  % (x)
        disp_path = "000%d_10.png"% (x)  

    imgL1 = cv2.imread(os.path.join(dataset_path, "left", im_path1))
    imgR1 = cv2.imread(os.path.join(dataset_path, "right", im_path1)) # stereo pair at t
    imgL2 = cv2.imread(os.path.join(dataset_path, "left", im_path2))
    imgR2 = cv2.imread(os.path.join(dataset_path, "right", im_path2)) # stereo pair at t+1
        
    input_data = np.concatenate((imgL1, imgR1, imgL2, imgR2), axis = 2)

    input_data = preprocess(input_data)
    xtrain.append(input_data)
    xtrain = np.array(xtrain)

    return xtrain, of_path, disp_path, disp_path