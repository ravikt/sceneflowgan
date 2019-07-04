import sys
sys.path.append("/home/traviku2/conditional_gan")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import create_generator, create_generatorBnormRes

#from viz.visualization import resultssf
from kitti_results import eval_results

from utils import datagen_eval

def kitti_eval():

    dataset_path = "/home/traviku2/kitti/test/"
    generator = create_generatorBnormRes()
    generator.load_weights('/home/traviku2/kitti/sfganC/generator_50_3.h5')

    for x in range(0,200): # total no. of scenes in KITTI
        input_blob, of, d1, d2 = datagen_eval(dataset_path, x)

        N = 0 # Sample number. There are 8 (0 to 7) samples in a batch
        stereo_images = input_blob[N, :, :, :]
        
        generated_sceneflow   = generator.predict(input_blob)
        generated_sceneflow   = generated_sceneflow[N, :, :, :]   

        # print input_blob.shape, groundtruth_sceneflow.shape
        eval_results(generated_sceneflow, of, d1, d2)

if __name__ == "__main__":
    kitti_eval()
