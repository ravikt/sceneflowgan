# Adversarial training for scene flow estimation
# Conditional Adversarial Network for Scene Flow

import keras 
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Conv2D, Add, UpSampling2D, \
     Cropping2D, Activation, MaxPooling2D, Flatten, Conv2DTranspose, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import multi_gpu_model


# image_dim = (384, 512, 3)
input_nc = 12
output_nc = 5
generator_input_dim = (384, 512, input_nc)
discriminator_input_dim = (384, 512, output_nc)

# Generates scene flow from stereo images
# The model will not be trained directly but
# It will be trained via GAN
# The Generator is a SceneEDNet, which takes pair 
# of stereo images and outputs scene flow

def create_generator():
    
    inimage = Input(shape=generator_input_dim)
    conv0 = Conv2D(64,   (3, 3), name = 'conv0',   strides = 2, padding='same')(inimage)
    conv0 = LeakyReLU()(conv0)
    conv1 = Conv2D(128,  (3, 3), name = 'conv1', strides = 2, padding='same')(conv0)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv2D(256,  (3, 3), name = 'conv2', strides = 2, padding='same')(conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv2D(512,  (3, 3), name = 'conv3', strides = 2, padding='same')(conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = Conv2D(1024, (3, 3), name = 'conv4', strides = 1, padding='same')(conv3)
    conv4 = LeakyReLU()(conv4) 

    conv5 = Conv2D(1024, (3, 3), name = 'conv5', strides = 1, padding='same')(conv4)
    conv5 = LeakyReLU()(conv5)
    up1   = UpSampling2D((2,2))(conv5)
    conv6 = Conv2D(512, (3, 3), name = 'conv6', strides = 1, padding='same')(up1)
    conv6 = LeakyReLU()(conv6)
    up2   = UpSampling2D((2,2))(conv6)
    conv7 = Conv2D(256, (3, 3), name = 'conv7', strides = 1, padding='same')(up2)
    conv7 = LeakyReLU()(conv7)
    up3   = UpSampling2D((2,2))(conv7)
    conv8 = Conv2D(128, (3, 3), name = 'conv8', strides = 1, padding='same')(up3)
    conv8 = LeakyReLU()(conv8)
    up4   = UpSampling2D((2,2))(conv8)
    #up4   = Cropping2D(cropping=((4,0),(0,0)))(up4)
    conv9 = Conv2D(64, (3, 3), name = 'conv9', strides = 1, padding='same')(up4)
    conv9 = LeakyReLU()(conv9)
    outimage = Conv2D(output_nc, (3, 3), name = 'output', strides = 1, padding='same')(conv9)
    #outimage = LeakyReLU()(outimage)

    generator = Model(inputs=inimage, outputs=outimage, name='Generator')
    # for multi-gpu
    generator = multi_gpu_model(generator, gpus=2)
    #generator.summary()

    return generator 

def create_generatorBnorm():

    inimage = Input(shape=generator_input_dim)
    conv0 = Conv2D(64,   (3, 3), name = 'conv0',   strides = 2, padding='same')(inimage)
    conv0 = LeakyReLU()(conv0)

    conv1 = Conv2D(128,  (3, 3), name = 'conv1', strides = 2, padding='same', use_bias=False)(conv0)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv2D(256,  (3, 3), name = 'conv2', strides = 2, padding='same', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2D(512,  (3, 3), name = 'conv3', strides = 2, padding='same', use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    conv4 = Conv2D(1024, (3, 3), name = 'conv4', strides = 1, padding='same', use_bias=False)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4) 

    conv5 = Conv2D(1024, (3, 3), name = 'conv5', strides = 1, padding='same', use_bias=False)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)
    

    up1   = UpSampling2D((2,2))(conv5)
    
    conv6 = Conv2D(512, (3, 3), name = 'conv6', strides = 1, padding='same', use_bias=False)(up1)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)
    
    up2   = UpSampling2D((2,2))(conv6)
    
    conv7 = Conv2D(256, (3, 3), name = 'conv7', strides = 1, padding='same', use_bias=False)(up2)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)

    up3   = UpSampling2D((2,2))(conv7)
    
    conv8 = Conv2D(128, (3, 3), name = 'conv8', strides = 1, padding='same', use_bias=False)(up3)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)

    up4   = UpSampling2D((2,2))(conv8)
    # up4   = Cropping2D(cropping=((4,0),(0,0)))(up4)
    
    conv9 = Conv2D(64, (3, 3), name = 'conv9', strides = 1, padding='same')(up4)
    conv9 = LeakyReLU()(conv9)

    output = Conv2D(output_nc, (3, 3), name = 'output', strides = 1, padding='same')(conv9)
    output = LeakyReLU()(output)
    model = Model(inputs=inimage, outputs=output)
    
    model.summary()
 
    return model

# Checks the differrence between the ground truth 
# scene flow and the one generated from generator
# The discriminator takes both real and fake samples
# and generates a probability
def create_discriminator():
    
    inimage = Input(shape=discriminator_input_dim)
    conv0 = Conv2D(128, (5,5), name='conv0', strides=2, padding='same')(inimage)
    conv0 = BatchNormalization()(conv0)
    conv0 = LeakyReLU()(conv0)

    conv1 = Conv2D(64, (5,5), name='conv1', strides=2, padding='same')(conv0)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    #conv2 = Conv2D(128, (5,5), name='conv2', strides=2)(conv1)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = LeakyReLU()(conv2)

    conv3 = Conv2D(64, (5,5), name='conv3', strides=2, padding='same')(conv1)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    #conv4 = Conv2D(32, (5,5), name='conv4', strides=2, padding='same')(conv3)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = LeakyReLU()(conv4)
    #conv1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(conv1)
    #conv1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(conv1)
    #conv1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(conv1)
    
    x = Flatten()(conv3)
    x = Dropout(0.4)(x)
    
    x = Dense(512, activation='tanh')(x)
    x = Dense(256, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    #x = Dense(64, activation='tanh')(x)
    # Output probability
    # Sigmoid tells us the probability if input sceneflow 
    # is real or not
    outprob = Dense(1, activation='sigmoid')(x)

    discriminator = Model(inputs=inimage, outputs=outprob, name='Discriminator')
    # for multi-gpu model
    discriminator = multi_gpu_model(discriminator, gpus=2)
    discriminator.summary()

    return discriminator 


def create_generatorBnormRes():

    inimage = Input(shape=generator_input_dim)

    conv0 = Conv2D(64,   (3, 3), name = 'conv0',   strides = 2, padding='same')(inimage)
    conv0 = BatchNormalization()(conv0)
    conv0 = LeakyReLU()(conv0)

    conv1 = Conv2D(128,  (3, 3), name = 'conv1', strides = 2, padding='same', use_bias=False)(conv0)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv2D(256,  (3, 3), name = 'conv2', strides = 2, padding='same', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2D(512,  (3, 3), name = 'conv3', strides = 2, padding='same', use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    conv4 = Conv2D(1024, (3, 3), name = 'conv4', strides = 1, padding='same', use_bias=False)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)

    #########################################

    conv5 = Conv2D(1024, (3, 3), name = 'conv5', strides = 1, padding='same', use_bias=False)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU()(conv5)

    skip1 = Add()([conv4, conv5]) # skiplayer-1 conv4 + conv5
    skip1 = LeakyReLU()(skip1)

    conv6 = Conv2D(512, (3, 3), name = 'conv6', strides = 1, padding='same', use_bias=False)(skip1)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU()(conv6)

    skip2 = Add()([conv3, conv6]) # skiplayer-2 conv3 + conv6
    skip2 = LeakyReLU()(skip2)
    up1   = UpSampling2D((2,2))(skip2) 
    
    conv7 = Conv2D(256, (3, 3), name = 'conv7', strides = 1, padding='same', use_bias=False)(up1)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU()(conv7)
   
    skip3 = Add()([conv2, conv7]) # skiplayer-3 conv2 + conv7
    skip3 = LeakyReLU()(skip3)
    up2   = UpSampling2D((2,2))(skip3)
    
    conv8 = Conv2D(128, (3, 3), name = 'conv8', strides = 1, padding='same', use_bias=False)(up2)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU()(conv8)

    skip4 = Add()([conv1, conv8]) # skiplayer-4 conv1 + conv8
    skip4 = LeakyReLU()(skip4)
    up3   = UpSampling2D((2,2))(skip4)
    
    conv9 = Conv2D(64, (3, 3), name = 'conv9', strides = 1, padding='same')(up3)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU()(conv9)

    skip5 = Add()([conv0, conv9]) # skiplayer-5 conv0 + conv9
    skip5 = LeakyReLU()(skip5)
    up4   = UpSampling2D((2,2))(skip5)


    output = Conv2D(output_nc, (3, 3), name = 'output', strides = 1, padding='same')(skip5)
    output = UpSampling2D((2,2))(output)
    
    model = Model(inputs=inimage, outputs=output)
    
    model.summary()
 
    return model

# The two models are combined together to make
# a GAN. The generator is trained via GAN, while discriminator
# is freezed. 
def get_gan_network(discriminator, generator):

    # discriminator.trainable = False

    gan_input = Input(shape=(384, 512, 12))
    # Fake scene flow generated by Generator
    # Takes consecutive streo pairs as input
    generated_sceneflow = generator(gan_input)
    # Error score for discriminator 
    gan_output = discriminator(generated_sceneflow)

    gan = Model(inputs=gan_input, outputs=[generated_sceneflow, gan_output])
    # for multi-gpu
    gan = multi_gpu_model(gan, gpus=2)

    return gan 


if __name__=='__main__':
    
    generator = create_generator()
    discriminator = create_discriminator()
    gan = get_gan_network(discriminator, generator)
