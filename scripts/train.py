import sys
sys.path.append("/home/ravi/Desktop/sceneflow/generative_model/second_chapter")

from tqdm import tqdm
import numpy as np
import os
import datetime
from keras import optimizers

from keras.utils import multi_gpu_model, plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History

from sfgan.model import create_generatorBnormRes, \
        create_discriminator, get_gan_network
from sfgan.utils import datagen
from sfgan.image_utils import writePFM
from sfgan.losses import epeloss, wasserstein_loss, joint_loss

learning_rate = 0.00001
nEpoch        = 100
train_set = 750# Number of training samples
lrdecay       = learning_rate/nEpoch
BASE_DIR = '/media/ravi/Data/Results/weights/'

def get_optimizer():
    return optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, 
           epsilon=None, decay=lrdecay, amsgrad=False)

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)



def train(epochs, train_folders):

    # Build the GAN network 
    adam = get_optimizer()
    generator = create_generatorBnormRes()
    discriminator = create_discriminator()
    gan = get_gan_network(discriminator, generator)


    # Compile the models
    discriminator.trainable = True
    discriminator.compile(optimizer=adam, loss=wasserstein_loss)
    discriminator.trainable = False
    loss = [epeloss, wasserstein_loss]
    loss_weights = [1, 1]
    gan.compile(optimizer=adam, loss=loss, loss_weights=loss_weights)
    #discriminator.trainable = True

    # Labels for generated and real scene flow
    # 8 is the batch size
    # y_gen = np.ones((8,1)) # Lable to train generator 
    #y_real = np.ones(8)
    y_real = np.ones((4,1))
    # label smoothing 
    y_real[:,0]=0.9 

    y_fake = -np.ones((4,1))
    

    for epoch in range(epochs):

        discriminator_losses = [] 
        gan_losses = []       

        # Introduces randomness in the input training data
        # Stochasiticity helps GAN to avoid getting stuck
        index = np.random.permutation(8) 
        # index = np.arange(8)

        print '-'*15, 'Epoch %d' % epoch, '-'*15
        for y in tqdm(range(train_folders)):
            # stereo_pairs - stack of input images
            # groundtruth_sceneflow - ground truth 
            stereo_pairs, groundtruth_sceneflow = datagen(y)
            # Get a "random" set of input images. These are the 

            for b in range(2):

                batch_index = index[b*4 : (b+1)*4]
                # Decode them to fake sceneflow 
                xtrain_images = stereo_pairs[batch_index]
                ytrain_sceneflow = groundtruth_sceneflow[batch_index]

                generated_sceneflow = generator.predict(xtrain_images)

                # print stereo_pairs.shape, groundtruth_sceneflow.shape, y_real.shape
                
                #for _ in range(2):
                    # Train discriminator ( Note : Could be trained multiple times)
                discriminator.trainable = True 
                d_loss_groundtruth = discriminator.train_on_batch(ytrain_sceneflow, y_real)
                d_loss_generated = discriminator.train_on_batch(generated_sceneflow, y_fake)
                d_loss = 0.5*d_loss_groundtruth + 0.5*d_loss_generated  
                discriminator_losses.append(d_loss)

                # Train generator only on discriminator's decidion and generated sceneflow
                discriminator.trainable = False
                gan_loss = gan.train_on_batch(xtrain_images, [ytrain_sceneflow, y_real]) # IS THIS CORRECT ?
                gan_losses.append(gan_loss)
                discriminator.trainable = True

        # Write Log
        print(np.nanmean(discriminator_losses), np.nanmean(gan_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.nanmean(discriminator_losses), np.nanmean(gan_losses)))
        
        save_all_weights(discriminator, generator, epoch, int(np.nanmean(gan_losses)))
        writePFM('of.pfm', generated_sceneflow[1,:,:,0:3])
        writePFM('d1.pfm', generated_sceneflow[1,:,:, 3])
        writePFM('d2.pfm', generated_sceneflow[1,:,:, 3])

        # Print metrics after every epoch
        # print('discriminator loss at epoch %s: %s' % (epoch, d_loss))
        # print('adversarial loss at epoch %s: %s' % (epoch, g_loss))
        
  
#plot_model(gan, to_file='model.png', show_shapes=False, 
#                        show_layer_names=True, rankdir='TB')

if __name__ == '__main__':
    train(nEpoch, train_set)
