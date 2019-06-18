import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")


import numpy as np
import matplotlib.pyplot as plt

from sfgan.model import create_generator, create_generatorBnormRes
from sfgan.image_utils import sceneflowconstruct, writePFM, readPFM

from viz.visualization import resultssf
from viz.ral import intro

from demo.utils import datagen

def results():

    predict_path = "/media/ravi/Data/SCENEFLOW/prediction_gan/Driving"
    input_blob, groundtruth_sceneflow = datagen(predict_path)

    generator = create_generatorBnormRes()
    generator.load_weights('/media/ravi/Data/Results/Thesis/RAL/sfganC/118/generator_50_3.h5')
    
    # extracting optical flow and depth maps from output
    N = 1 # Sample number. There are 8 samples in a batch
    stereo_images = input_blob[N, :, :, :]

    generated_sceneflow   = generator.predict(input_blob)
    generated_sceneflow   = generated_sceneflow[N, :, :, :]
    groundtruth_sceneflow = groundtruth_sceneflow[N, :, :, :]

    #print stereo_images.shape
    #print generated_sceneflow.shape
    intro(stereo_images, groundtruth_sceneflow, generated_sceneflow)


def kitti():

    predict_path = "/media/ravi/Data/SCENEFLOW/prediction_gan/KITTI"
    stereo_pairs, scene_flow = datagen(predict_path)

    print stereo_pairs.shape

if __name__ == "__main__":
    kitti()
