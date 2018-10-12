import keras
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Input, UpSampling2D, BatchNormalization, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, Conv2DTranspose, GlobalMaxPool2D, Dropout, Reshape
from keras.optimizers import Adam

class Adversarial_Model(object):
    def __init__(self):
        self.D = Sequential()
        self.G = Sequential()
        self.create_d()
        self.create_g()
        self.GAN = Sequential()
        self.GAN.add(self.G)
        self.GAN.add(self.D)
        Dopt = Adam(lr=1e-4)
        Gopt = Adam(lr=1e-3)
        self.D.compile(optimizer='adagrad', loss='categorical_crossentropy',
                        metrics=['accuracy'])
        self.GAN.compile(optimizer='adagrad', loss='categorical_crossentropy',
                        metrics=['accuracy'])

    def create_d(self):
        self.D.add(Conv2D(16, 7, strides=2, activation='elu', input_shape=(224, 224, 3)))
        self.D.add(Conv2D(16, 7, strides=2, activation='elu'))
        self.D.add(Dropout(0.5))
        self.D.add(Conv2D(32, 5, strides=2, activation='elu'))
        self.D.add(Conv2D(32, 5, strides=2, activation='elu'))
        self.D.add(Dropout(0.5))
        self.D.add(Conv2D(64, 3, activation='elu'))
        self.D.add(Conv2D(64, 3, activation='elu'))
        self.D.add(Flatten())
        self.D.add(Dense(2, activation='sigmoid'))

    def create_g(self):
        self.G.add(Dense(28*28*12, activation='elu', input_shape=(100,)))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(Reshape((28, 28, 12)))
        self.G.add(Conv2DTranspose(8, 3, padding='same', activation='elu'))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(6, 3, padding='same', activation='elu'))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(3, 3, padding='same', activation='elu'))
        self.G.add(BatchNormalization())
        self.G.add(Dropout(0.5))
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(3, 3, padding='same', activation='sigmoid'))