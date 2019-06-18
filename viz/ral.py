import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")

import datetime
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from viz.colorcode import flow_to_color

from sfgan.image_utils import readPFM
from sfgan.image_utils import overlay  

now = datetime.datetime.now()

def intro(stereo_image, gt_sceneflow, gen_sceneflow):

    ###### Gets the predicted and ground truth scene flow with stereo images

    optical_flow = gen_sceneflow[:, :, 0:3]  #
    disparity1   = gen_sceneflow[:, :, 3]    # Predictions
    disparity2   = gen_sceneflow[:, :, 4]    #

    optical_flow = np.array(optical_flow)

    u = optical_flow[:,:,0]
    v = optical_flow[:,:,1]

    pred_flow = np.dstack((u,v))  # Predicted optical flow
    pred_flow = flow_to_color(pred_flow, convert_to_bgr=False)

    gt_optical_flow = gt_sceneflow[:, :, 0:3] #
    gt_disparity1   = gt_sceneflow[:, :, 3]   # Ground Truth
    gt_disparity2   = gt_sceneflow[:, :, 4]   #

    gt_u = gt_optical_flow[:, :, 0] 
    gt_v = gt_optical_flow[:, :, 1]

    gt_flow = np.dstack((gt_u, gt_v)) # Ground truth opticla flow
    gt_flow = flow_to_color(gt_flow, convert_to_bgr=False)

    imL1 = stereo_image[:,:,0:3]
    imL2 = stereo_image[:,:,6:9]

    left_overlay = overlay(imL1, imL2, 0.5)

    imR1 = stereo_image[:,:,3:6] 
    imR2 = stereo_image[:,:,9:12]

    right_overlay = overlay(imR1, imR2, 0.5)

    ####### create subplot

    '''
    plt.figure()
    G1 = gridspec.GridSpec(1, 2)
    G1.update(left=0.05, right=0.48, wspace=0.05)

    ax1 = plt.subplot(G1[0, 0])
    ax1.axis('off')
    ax1.imshow(left_overlay)
  
    ax2 = plt.subplot(G1[0, 1])
    ax2.axis('off')
    ax2.imshow(right_overlay)
    
    G2 = gridspec.GridSpec(2, 3)
    G2.update(left=0.55, right=0.98, hspace=0.05)
    ax3 = plt.subplot(G2[0,0])
    ax3.axis('off')
    ax3.imshow(gt_flow)

    ax4 = plt.subplot(G2[0,1])
    ax4.axis('off')
    ax4.imshow(gt_disparity1, cmap='gray')

    ax5 = plt.subplot(G2[0,2])
    ax5.axis('off')
    ax5.imshow(gt_disparity2, cmap='gray')

    ax6 = plt.subplot(G2[1,0])
    ax6.axis('off')
    ax6.imshow(pred_flow)

    ax7 = plt.subplot(G2[1,1])
    ax7.axis('off')
    ax7.imshow(disparity1, cmap='gray')

    ax8 = plt.subplot(G2[1,2])
    ax8.axis('off')
    ax8.imshow(disparity2, cmap='gray')

    plt.tight_layout()

    plt.savefig('../samples/intro_{}_{}.png'.format(now.month, now.day), set_dpi=1000)
    '''

    # Display the image

    plt.figure()
    plt.axis('off')
    plt.imshow(left_overlay)
    plt.savefig('left.png', transparent= True, bbox_inches='tight')

    plt.figure()
    plt.axis('off')
    plt.imshow(right_overlay)
    plt.savefig('right.png', transparent= True, bbox_inches='tight')

    plt.figure()
    plt.axis('off')
    plt.imshow(pred_flow)
    plt.savefig('sf_flow_pr.png', transparent= True, bbox_inches='tight')
    
    plt.figure()
    plt.axis('off')
    plt.imshow(gt_flow)
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
   