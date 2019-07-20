import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")


import numpy as np
import matplotlib.pyplot as plt

from sfgan.model import create_generator, create_generatorBnormRes
from sfgan.utils import datagen
from sfgan.image_utils import sceneflowconstruct, writePFM, readPFM

from viz.visualization import resultssf
from viz.predict import predict_sceneflow

def test():

    folder_number = 25 #51
    stereo_pairs, scene_flow = datagen(folder_number)

    # for the bacth size 4
    stereo_pairs = stereo_pairs[0:4]

    generator = create_generatorBnormRes()
    generator.load_weights('/media/ravi/Data/Results/ral/sfganC/118/generator_70_2.h5')
    
    # extracting optical flow and depth maps from output
    N = 1 # Sample number. There are 8 samples in a batch
    stereo_images = stereo_pairs[N, :, :, :]
    scene_flow   = scene_flow[N, :, :, :]

    generated_sceneflow = generator.predict(stereo_pairs)
    groundtruth_sceneflow = scene_flow
    generated_sceneflow = generated_sceneflow[N, :, :, :]

    print stereo_images.shape
    print generated_sceneflow.shape

    #resultssf(stereo_images, groundtruth_sceneflow, generated_sceneflow)
    #resultssf(groundtruth_sceneflow, generated_sceneflow)
    predict_sceneflow(stereo_images, groundtruth_sceneflow, generated_sceneflow)

'''
    for i in range(3):
        fig = plt.figure(figsize=[8,6])
        plt.axis('off')
        predFlow = generated_sceneflow[:,:,i]
        # This part removes the white spaces
        ax = fig.add_subplot(111)
        ax.imshow(predFlow)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plt.savefig('sf'+str(i)+'pr.png',transparent = True, bbox_inches='tight', pad_inches = 0)
        plt.clf()
        
        fig = plt.figure(figsize=[8,6])
        plt.axis('off')
        gtFlow = groundtruth_sceneflow[:,:,i]
        ax = fig.add_subplot(111)
        ax.imshow(gtFlow)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plt.savefig('sf'+str(i)+'gt.png',transparent = True, bbox_inches='tight', pad_inches = 0) 
        plt.clf()
'''

if __name__ == "__main__":
    test()
