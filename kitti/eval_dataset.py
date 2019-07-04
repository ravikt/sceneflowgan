# Code to resize the entire KITTI dataset

import numpy as np
import cv2
import os
import re

from image_utils import kitti_disp_read, kitti_flow_read, writeImage\
                        preprocess, kitti_disp_write, kitti_flow_write
from utils import datagen 

dataset_path = "/home/traviku2/kitti_scene_flow/training/"
output_path  = "/home/traviku2/"



# for output
gt_img_dir = "data/scene_flow/image_2"
gt_obj_map_dir = "data/scene_flow/obj_map"
gt_disp_noc_0_dir = "data/scene_flow/disp_noc_0"
gt_disp_occ_0_dir = "data/scene_flow/disp_occ_0"
gt_disp_noc_1_dir = "data/scene_flow/disp_noc_1"
gt_disp_occ_1_dir = "data/scene_flow/disp_occ_1"
gt_flow_noc_dir = "data/scene_flow/flow_noc"
gt_flow_occ_dir = "data/scene_flow/flow_occ"

for x in range(200):

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

    img10 = cv2.imread(os.path.join(dataset_path, "image_2",im_path1))
    img11 = cv2.imread(os.path.join(dataset_path, "image_2",im_path2))
    
    of_noc = kitti_flow_read(os.path.join(dataset_path, of_dir, of_path))
    of_occ = kitti_flow_read(os.path.join(dataset_path, of_dir, of_path))

     kitti_disp_read(os.path.join(dataset_path, d_dir,"disp_0", disp_path1))
     