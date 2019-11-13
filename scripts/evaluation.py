import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")


import numpy as np
from sfgan.model import create_generatorBnormRes
from sfgan.utils import datagen
from sfgan.losses import eval_joint_loss
 
def evaluate():

    folder_number = 150
    flow_epe = []
    d1_err = []
    d2_err = []

    generator = create_generatorBnormRes()

    generator.load_weights('/media/ravi/Data/Results/ral/sfganC/117/generator_30_3.h5')
    
    for serial_num in range(folder_number):

        stereo_pairs, scene_flow = datagen(serial_num)

        # for the bacth size 4
        stereo_pairs = stereo_pairs[0:4]
        
        '''
        # extracting optical flow and depth maps from output
        N = 1 # Sample number. There are 8 samples in a batch
        stereo_images = stereo_pairs[N, :, :, :]
        scene_flow   = scene_flow[N, :, :, :]
        '''
        generated_sceneflow = generator.predict(stereo_pairs)
        groundtruth_sceneflow = scene_flow[0:4]

        #print generated_sceneflow.shape, groundtruth_sceneflow.shape

        for b in range(4):
        
            pred_sceneflow = generated_sceneflow[b, :, :, :]
            gt_sceneflow = scene_flow[b,:,:,:]
            x,y,z = eval_joint_loss(gt_sceneflow, pred_sceneflow)

            flow_epe.append(x)
            d1_err.append(y)
            d2_err.append(z)

            #print serial_num, b, x, y, z      
            with open('eval-C30-A.txt', 'a+') as f:
                f.write('{} - {} - {} - {} - {} - {}\n'.format(np.median(flow_epe), np.mean(flow_epe), np.median(d1_err), np.mean(d1_err), np.median(d2_err), np.mean(d2_err)))
                    
        print serial_num
    print np.median(flow_epe), np.mean(flow_epe), np.median(d1_err), np.mean(d1_err), np.median(d2_err), np.mean(d2_err)

if __name__ == "__main__":
    evaluate()
