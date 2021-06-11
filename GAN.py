import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from pathlib import Path
from scipy import ndimage
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
import clustering
import transformations


class GAN():
    def __init__(self):
        self.img_rows = 500
        self.img_columns = 500
        #self.channels = 3
        self.img_shape = (self.img_rows, self.img_columns)#, self.channels)
        self.latent_dim = 250

        optimizer = Adam(0.00025, 0.5)  # learning rate of 0.00025

        # create the Generator and Discriminator
        self.generator = self.define_generator()
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # loss function to optimise G is min(log(1-D)), in practise we use max(log(D))
        # need to implement this loss
        self.discriminator = self.define_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator.trainable = False  # only training generator

        valid_input = self.discriminator(img)  # determine whether image is real or fake

        # overall GAN with generator and discriminator combined
        self.combined = Model(noise, valid_input)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def define_generator(self):
        model = Sequential()

        model.add(Dense(32, input_shape=(self.latent_dim, )))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        #
        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.9))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim, ))
        generated_image = model(noise)

        return Model(noise, generated_image)

    def define_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))  # Flattens input without affecting batch size

        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        #
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        discriminated_image = Input(shape=self.img_shape)
        valid = model(discriminated_image)

        return Model(discriminated_image, valid)

    def train(self, cloud_images, epochs, batch_size, interval, model_path, save_interim_images_path):
        # normalisation of images between [-1, 1]
        # using formula: 2 * (x - x.min()) / (x.max() - x.min()) - 1
        # x.min() being 0, and x.max() being 255 for RGB images
        cloud_images = np.array(cloud_images)
        cloud_images = cloud_images.astype('float32')
        cloud_images = cloud_images/127.5 - 1

        num_train = cloud_images.shape[0]

        real = np.ones((batch_size, 1))  # ground truth array of 1's for real, 0's for fake
        fake = np.zeros((batch_size, 1))
        total_batches = 0

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            num_batches = int(np.ceil(num_train/batch_size))
            progress_bar = Progbar(target=num_batches)

            for i in range(num_batches):
                real_image_indices = np.random.randint(0, cloud_images.shape[0], batch_size)
                real_image_batch = cloud_images[real_image_indices]

                # generate images to try and fool the discriminator with
                disc_training_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                disc_generated_images = self.generator.predict(disc_training_noise)
                # Train discriminator
                disc_loss_real = self.discriminator.train_on_batch(real_image_batch, real)
                disc_loss_fake = self.discriminator.train_on_batch(disc_generated_images, fake)
                
                total_discriminator_loss = np.add(disc_loss_real, disc_loss_fake) / 2
                # Train Generator in the combined model - because self.discriminator.trainable is set to False
                gen_training_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_loss = self.combined.train_on_batch(gen_training_noise, real)

                print("\nEpoch:", (epoch+1), "\nDiscriminator Loss:", total_discriminator_loss[0],
                      "\nAccuracy:", total_discriminator_loss[1] * 100, "\nGenerator Loss:", gen_loss)

                if total_batches % interval == 0:
                    self.generate_and_save_images(total_batches, save_interim_images_path)

                total_batches = total_batches + 1
                progress_bar.update(i+1)
        
        # Added code to save trained discriminator and generator models
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.generator.save_weights(model_path + "generatorWeights.h5")
        self.discriminator.save_weights(model_path + "discriminatorWeights.h5")
        '''
        # Added code to generate some images when all epochs are completed...
        gen_noise = np.random.normal(0, 1, (num_gen_images, self.latent_dim))
        gen_images = self.generator.predict(gen_noise)
        print(gen_images.shape)
        for i in range(num_gen_images):
            imageI = gen_images[i,:,:]
            imageI = imageI * 127.5 + 127.5
            im = Image.fromarray(imageI)
            im = im.convert("L")
            im.save("post_training_generation/%d.png" % (i+1))
        '''

    def generate_and_save_images(self, epoch, save_path):
        row_images, column_images = 1, 1  # 4, 4
        # number of images we generate is equal to --> row_images * column_images
        gen_noise = np.random.normal(0, 1, (row_images*column_images, self.latent_dim))
        gen_images = self.generator.predict(gen_noise)
        
        '''
        New code added here
        '''
        # rescaling images - removed the step of "/2 + 0.5" - made no sense to me
        # Since generated image is in range [-1, 1], rescaling only requires to multiply by 127.5 ([-127.5, 127.5]) and then adding 127.5 ([0, 255])
        gen_images = gen_images[0,:,:] * 127.5 + 127.5 # Only first image taken because only one is generated at a time
        # Save images in actual resolution, i.e. (500X500) - so that they are usable for training
        im = Image.fromarray(gen_images.astype('uint8'))
        im = im.convert("L")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        im.save(save_path + str(epoch) + '.png')
    
    def load_model_and_generate_images(self, model_path, save_path, num_gen_images):
        self.generator.load_weights(model_path + "generatorWeights.h5")
        self.discriminator.load_weights(model_path + "discriminatorWeights.h5")
        gen_noise = np.random.normal(0, 1, (num_gen_images, self.latent_dim))
        gen_images = self.generator.predict(gen_noise)
        print(gen_images.shape)
        for i in range(num_gen_images):
            imageI = gen_images[i,:,:]
            imageI = imageI * 127.5 + 127.5
            im = Image.fromarray(imageI.astype('uint8'))
            im = im.convert("L")
            Path(save_path).mkdir(parents=True, exist_ok=True)
            im.save(save_path + str(i+1) + '.png')


def train_GAN(source_images_path, model_path, save_interim_images_path):
    print("Loading data to train GAN")
    transformed_cloud_images = transformations.data_load(source_images_path)
    print("Number of loaded images: ", len(transformed_cloud_images))
    print("Size of first loaded image: ", transformed_cloud_images[0].shape)
    
    random.shuffle(transformed_cloud_images)
    
    gan = GAN()
    gan.train(transformed_cloud_images, epochs=1000, batch_size=32, interval=10, model_path = model_path, save_interim_images_path = save_interim_images_path)
    '''
    with tf.device("/cpu:0"):
        gan = GAN()
        gan.train(transformed_cloud_images, epochs=1000, batch_size=32, interval=10)
    '''

def load_generate_GAN(model_path, save_path, num_gen_images = 100):
    gan = GAN()
    gan.load_model_and_generate_images(model_path, save_path, num_gen_images)

def r_b_channel_conversion(image_path, save_path):
    cloud_images = transformations.data_load(image_path)
    r_b_image = cloud_images
    for (idx,image) in enumerate(cloud_images):
        temp = image[:,:,0].astype('float64') - image[:,:,2].astype('float64') # Extract R-B channel
        #temp = ((temp+255)/2).astype('uint8') # Rearrange the values in range [0, 255] and make it of type 'uint8'
        temp = (((temp - np.min(temp))/(np.max(temp) - np.min(temp)))*255).astype('uint8') # Rearrange the values in range [0, 255] and make it of type 'uint8'
        r_b_image[idx] = temp
        im = Image.fromarray(r_b_image[idx])
        im = im.convert("L")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        im.save(save_path + str(idx+1) + '.png')

def transform_images(image_path, save_path):
    # Used to perform the various affine transformations on the image dataset
    cloud_images = transformations.data_load(image_path)
    clouds_90, clouds_180, clouds_270, clouds_rotated_overall = transformations.rotation(cloud_images)
    overall_cloud_images = transformations.reflection(clouds_rotated_overall)
    transformations.save_images(overall_cloud_images, save_path)
    
def binary_maps(source_images_path, save_path):
    file_list = [f for f in os.listdir(source_images_path) if os.path.isfile(os.path.join(source_images_path, f))]
    for file in file_list:
        img_num = int(file.split('.')[0])
        image_location = source_images_path + str(img_num) + ".png"
        clustering.run_kmeans(image_location, img_num, save_path)

if __name__ == '__main__':
    ori_image_path = './data/images/'
    ori_GT_binMap_path = './data/GTmaps/'
    
    r_b_channel_path = './data/R-B_Original/'
    transformed_image_path = './data/transformed_images/'
    train_gan_gen_images = './data/training_generated_images/'
    final_gan_gen_images = './data/final_generated_images/'
    binMaps_final_gen_images = './data/BINmaps_generated_images/'
    
    model_save_path = './models/'
    
    # Used to create R-B channel from original image
    #r_b_channel_conversion(ori_image_path, r_b_channel_path)
    # Used to perform the various affine transformations on the image dataset
    #transform_images(r_b_channel_path, transformed_image_path)
    
    # training the GAN and save the model
    print("Calling Training Method...")
    train_GAN(source_images_path = transformed_image_path, model_path = model_save_path, save_interim_images_path = train_gan_gen_images)
    print("GAN Trained!\n")
    
    # loading trained weights and generating final images
    print("Calling Loading and Generating Method...")
    load_generate_GAN(model_save_path, final_gan_gen_images, num_gen_images = 100)
    print("Final Images Generated!\n")

    # used to generate the binary maps for a sample of images
    print("Creating Binary Maps of Generated Images...")
    binary_maps(final_gan_gen_images, binMaps_final_gen_images)
    print("Binary Maps Created!\n")