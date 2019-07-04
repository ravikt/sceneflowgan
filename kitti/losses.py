# For performing true division
from __future__ import division

import numpy as np 
import keras
import keras.backend as K
from keras.models import Model 


def epeloss(y_true, y_pred):
    x = y_true[:,:,1] - y_pred[:,:,1]
    y = y_true[:,:,2] - y_pred[:,:,2]
    z = y_true[:,:,3] - y_pred[:,:,3]
    loss = K.square(x) + K.square(y) + K.square(z)
    loss = K.sqrt(loss)
    return K.mean(loss) 

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def joint_loss(y_true, y_pred):
    
    # Optical flow error
    u_error  = K.square(y_true[:,:,0] - y_pred[:,:,0])
    v_error  = K.square(y_true[:,:,1] - y_pred[:,:,1])
    
    # Disparity error
    d1_error  = y_true[:,:,3] - y_pred[:,:,3]
    d2_error  = y_true[:,:,4] - y_pred[:,:,4]

    # End point error for opticla flow and disparity 
    epe_opticalflow = 0.333 * K.sqrt(u_error + v_error)
    epe_disparity   = 0.333 * K.abs(d1_error) + 0.333 * K.abs(d2_error)

    # Joint loss function
    loss = epe_opticalflow + epe_disparity

    return K.mean(loss) 


def eval_joint_loss(y_true, y_pred):
    
    # Optical flow error
    u_error  = np.square(y_true[:,:,0] - y_pred[:,:,0])
    v_error  = np.square(y_true[:,:,1] - y_pred[:,:,1])
    
    # Disparity error
    d1_error  = y_true[:,:,3] - y_pred[:,:,3]
    d2_error  = y_true[:,:,4] - y_pred[:,:,4]

    # End point error for opticla flow and disparity 
    epe_opticalflow =  np.sqrt(u_error + v_error)
    disp1           =  np.absolute(d1_error) 
    disp2           =  np.absolute(d2_error)

    # Joint loss function
    # loss = epe_opticalflow + epe_disparity

    return np.nanmean(epe_opticalflow), np.nanmean(disp1), np.nanmean(disp2)
