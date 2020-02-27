import os
import glob
import random
import subprocess
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Dense
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.applications import VGG19

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xzf - -C data", shell=True)

"""
    Implement model with General Adversarial Network Architecture
"""
class SUPER_RESOLUTION_PIXELS():
    def __init__(self):
        self.num_epochs = 20
        self.batch_size = 32
        self.channels = 3
        self.input_height = 32
        self.input_width = 32
        self.output_height = 256
        self.output_width = 256
        self.lr_shape = (self.input_height, self.input_width, self.channels)
        self.hr_shape = (self.output_height, self.output_width, self.channels)
        self.val_dir = 'data/test'
        self.train_dir = 'data/train'

        self.disFilters = 64
        self.genFilters = 64
        self.n_residual_blocks = 8

        self.num_steps_per_epoch = len(
            glob.glob(self.train_dir + "/*-in.jpg")) // self.batch_size
        self.val_steps_per_epoch = len(
            glob.glob(self.val_dir + "/*-in.jpg")) // self.batch_size

        self.val_generator = self.image_generator(self.batch_size, self.val_dir)
        self.in_sample_images, self.out_sample_images = next(self.val_generator)

    def buildModelOnConvUpSampling(self):
        self.model = Sequential()
        self.model.add(Conv2D(3, (3, 3), activation='relu', padding='same',
                                input_shape=(self.input_width, self.input_height, self.channels)))
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

        self.model.compile(optimizer='adam', loss='mse',
                      metrics=[self.perceptual_distance])

    def build_model(self):
        # basic model with Conv & UpSample
        self.buildModelOnConvUpSampling()

        # Below code represents GAN Architecture
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        self.discriminator = self.build_discriminator()
        self.dicrimator.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.generator = self.build_generator()

        image_hr = Input(shape=self.hr_shape)
        image_lr = Input(shape=self.lr_shape)

        generated_hr = self.generator(image_lr)
        generated_features = self.vgg(generated_hr)

        self.discriminator.trainable = False

        validity = self.discriminator(generated_hr)
        self.combined_model = Model([image_lr, image_hr], [validity, generated_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights = [le-3, 1],
                              optimizer=Adam(0.0002, 0.5))


    def build_vgg(self):
        # Pre-trained VGG19 model to extract image features
        vgg = VGG19(weights='imagenet')
        img = Input(shape = self.hr_shape)
        img_features = vgg(img)

        return Model(img, img_features)

    def train(self):
        self.model.fit_generator(self.image_generator(self.batch_size, self.train_dir),
                            steps_per_epoch=self.num_steps_per_epoch,
                            epochs=self.num_epochs,
                            validation_steps=self.val_steps_per_epoch,
                            validation_data=self.val_generator)

        self.model.save('sres_model.h5')

    def build_generator(self):
        """
            Takes input layer as low resolution image(img_lr) and generates high
            resolution image(gen_hr).
        """

        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deConv2d(layer_input):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        img_lr = Input(shape=self.lr_shape)
        g1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        g1 = Activation('relu')(g1)

        r = residual_block(c1, self.genFilters)
        for i in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.genFilters)

        g2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Add()([g2, g1])

        u1 = deConv2d(g2)
        u2 = deConv2d(u1)

        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):
        def d_block(layer_input, filters, strides=1, batchNormal=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if batchNormal:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input image
        d0 = Input(shape=self.hr_shape)
        d1 = d_block(d0, self.disFilters, batchNormal=False)
        d2 = d_block(d1, self.disFilters, strides=2)
        d3 = d_block(d2, self.disFilters * 2)
        d4 = d_block(d3, self.disFilters * 2, strides=2)
        d5 = d_block(d4, self.disFilters * 4)
        d6 = d_block(d5, self.disFilters * 4, strides=2)
        d7 = d_block(d6, self.disFilters * 8)
        d8 = d_block(d7, self.disFilters * 8, strides=2)

        d9 = Dense(self.disFilters*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d10, validity)

    def image_generator(self, nbatch_size, img_dir):
        """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
        input_filenames = glob.glob(img_dir + "/*-in.jpg")
        counter = 0
        random.shuffle(input_filenames)
        while True:
            small_images = np.zeros(
                (nbatch_size, self.input_width, self.input_height, 3))
            large_images = np.zeros(
                (nbatch_size, self.output_width, self.output_height, 3))
            if counter+nbatch_size >= len(input_filenames):
                counter = 0
            for i in range(nbatch_size):
                img = input_filenames[counter + i]
                small_images[i] = np.array(Image.open(img)) / 255.0
                large_images[i] = np.array(
                    Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
            yield (small_images, large_images)
            counter += nbatch_size

    def perceptual_distance(self, y_true, y_pred):
        """Calculate perceptual distance, DO NOT ALTER"""
        y_true *= 255
        y_pred *= 255
        rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
        r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
        g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
        b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

        return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


def main():
    sr_pixels = SUPER_RESOLUTION_PIXELS()
    sr_pixels.build_model()
    #sr_pixels.train()

if __name__ == "__main__":
    main()
