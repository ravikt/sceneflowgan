# Code to visualize Optical flow and Disparity 
# 
# Adapted from 
# Author: Tom Runia
# Date Created: 2018-08-03

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")

import datetime
import numpy as np
import matplotlib.pyplot as plt
from viz.colorcode import flow_to_color

now = datetime.datetime.now()

def results(stereo_image, gt_sceneflow, gen_sceneflow):

    optical_flow = gen_sceneflow[:, :, 0:3]
    disparity1   = gen_sceneflow[:, :, 3]  
    disparity2   = gen_sceneflow[:, :, 4]

    optical_flow = np.array(optical_flow)

    u = optical_flow[:,:,0]
    v = optical_flow[:,:,1]

    pred_flow = np.dstack((u,v))

    #print pred_flow.shape 

    gt_optical_flow = gt_sceneflow[:, :, 0:3]
    gt_disparity1   = gt_sceneflow[:, :, 3]
    gt_disparity2   = gt_sceneflow[:, :, 4]

    gt_u = gt_optical_flow[:, :, 0] 
    gt_v = gt_optical_flow[:, :, 1]

    gt_flow = np.dstack((gt_u, gt_v))

    # Apply the coloring (for OpenCV, set convert_to_bgr=True)
    flow_color = flow_to_color(pred_flow, convert_to_bgr=False)

    gt_flow_color = flow_to_color(gt_flow, convert_to_bgr=False)

    # Display the image

    fig = plt.figure()

    ax1 = fig.add_subplot(3,2,1)
    ax1 = plt.axis('off')
    ax1 = plt.imshow(flow_color)
    ax1 = plt.savefig('sf_flow_pr.png', transparent= True, bbox_inches='tight')
    
    ax2 = fig.add_subplot(3,2,2)
    ax2 = plt.axis('off')
    ax2 = plt.imshow(gt_flow_color)
    ax2 = plt.savefig('sf_flow_gt.png', transparent= True, bbox_inches='tight')

    ax3 = fig.add_subplot(3,2,3)
    ax3 = plt.axis('off')
    ax3 = plt.imshow(disparity1, cmap='gray')
    ax3 = plt.savefig('sf_d1_pr.png', transparent= True, bbox_inches='tight')

    ax4 = fig.add_subplot(3,2,4)
    ax4 = plt.axis('off')
    ax4 = plt.imshow(gt_disparity1, cmap='gray')
    ax4 = plt.savefig('sf_d1_gt.png', transparent= True, bbox_inches='tight')
 

    ax5 = fig.add_subplot(3,2,5)
    ax5 = plt.axis('off')
    ax5 = plt.imshow(disparity2, cmap='gray')
    ax5 = plt.savefig('sf_d2_pr.png', transparent= True, bbox_inches='tight')

    ax6 = fig.add_subplot(3,2,6)
    ax6 = plt.axis('off')
    ax6 = plt.imshow(gt_disparity2, cmap='gray')
    ax6 = plt.savefig('sf_d2_gt.png', transparent= True, bbox_inches='tight')
    #fig.savefig('../samples/sfgan_{}_{}.png'.format(now.month, now.day), set_dpi=500)
    plt.show()


def vis_opticalflow(of):

    u = of[:,:,0]
    v = of[:,:,1]

    pred_flow = np.dstack((u,v))
    flow_color = flow_to_color(pred_flow, convert_to_bgr=False)

    plt.figure()
    plt.imshow(flow_color)
    plt.show()


def resultssf(stereo_image, gt_sceneflow, gen_sceneflow):

    optical_flow = gen_sceneflow[:, :, 0:3]
    disparity1   = gen_sceneflow[:, :, 3]  
    disparity2   = gen_sceneflow[:, :, 4]

    optical_flow = np.array(optical_flow)

    u = optical_flow[:,:,0]
    v = optical_flow[:,:,1]

    pred_flow = np.dstack((u,v))

    #print pred_flow.shape 

    gt_optical_flow = gt_sceneflow[:, :, 0:3]
    gt_disparity1   = gt_sceneflow[:, :, 3]
    gt_disparity2   = gt_sceneflow[:, :, 4]

    gt_u = gt_optical_flow[:, :, 0] 
    gt_v = gt_optical_flow[:, :, 1]

    gt_flow = np.dstack((gt_u, gt_v))

    # Apply the coloring (for OpenCV, set convert_to_bgr=True)
    flow_color = flow_to_color(pred_flow, convert_to_bgr=False)

    gt_flow_color = flow_to_color(gt_flow, convert_to_bgr=False)

    # Display the image
    plt.figure()
    plt.axis('off')
    plt.imshow(flow_color)
    plt.savefig('sf_flow_pr.png', transparent= True, bbox_inches='tight')
    
    plt.figure()
    plt.axis('off')
    plt.imshow(gt_flow_color)
    plt.savefig('sf_flow_gt.png', transparent= True, bbox_inches='tight')

    plt.figure()
    plt.axis('off')
    plt.imshow(disparity1, cmap='gray')
    plt.savefig('sf_d1_pr.png', transparent= True, bbox_inches='tight')
 
    plt.figure()
    plt.axis('off')
    plt.imshow(gt_disparity1, cmap='gray')
    plt.savefig('sf_d1_gt.png', transparent= True, bbox_inches='tight')
 
    plt.figure()
    plt.axis('off')
    plt.imshow(disparity2, cmap='gray')
    plt.savefig('sf_d2_pr.png', transparent= True, bbox_inches='tight')

    plt.figure()
    plt.axis('off')
    plt.imshow(gt_disparity2, cmap='gray')
    plt.savefig('sf_d2_gt.png', transparent= True, bbox_inches='tight')
    #fig.savefig('../samples/sfgan_{}_{}.png'.format(now.month, now.day), set_dpi=500)
    
    
    plt.show()
