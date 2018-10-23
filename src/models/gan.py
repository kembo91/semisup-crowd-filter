import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Input, UpSampling2D, BatchNormalization, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, Conv2DTranspose, GlobalMaxPool2D, Dropout, Reshape
from keras.optimizers import Adam

class Adversarial_Model(object):
    def __init__(self):
        self.D = Sequential()
        self.G = Sequential()
        Dopt = Adam(lr=1e-5)
        Gopt = Adam(lr=1e-4)
        self.create_d()
        self.create_g()
        self.D.compile(optimizer=Dopt, loss='binary_crossentropy')
        self.GAN = Sequential()
        self.GAN.add(self.G)
        self.GAN.add(self.D)
        self.GAN.layers[1].trainable = False
        self.GAN.compile(optimizer=Gopt, loss='binary_crossentropy')

    def create_d(self):
        self.D.add(Conv2D(64, 5, strides=2, activation='elu', input_shape=(224, 224, 3)))
        self.D.add(BatchNormalization())
        self.D.add(Conv2D(128, 5, strides=2, activation='elu'))
        self.D.add(BatchNormalization())
        self.D.add(Dropout(0.5))
        self.D.add(Conv2D(256, 5, strides=2, activation='elu'))
        self.D.add(BatchNormalization())
        self.D.add(Conv2D(512, 5, strides=2, activation='elu'))
        self.D.add(BatchNormalization())
        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))

    def create_g(self):
        self.G.add(Dense(7*7*256, activation='elu', input_shape=(100,)))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(Reshape((7, 7, 256)))
        self.G.add(Conv2DTranspose(256, 3, strides=2, activation='elu', padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(Conv2DTranspose(256, 3, strides=2, activation='elu', padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Conv2DTranspose(256, 3, strides=2, activation='elu', padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(Conv2DTranspose(256, 3, strides=1, activation='elu', padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Conv2DTranspose(128, 3, strides=2, activation='elu', padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(Conv2DTranspose(64, 3, strides=2, activation='elu', padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Conv2DTranspose(3, 3, strides=1, activation='sigmoid', padding='same'))