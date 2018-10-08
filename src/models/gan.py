import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, BatchNormalization, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, Conv2DTranspose, GlobalMaxPool2D, Dropout, Reshape

class Adversarial_Model(object):
    def __init__(self):
        d_input = Input(shape=(224, 224, 3))
        g_input = Input(shape=(100,))
        gan_input = Input(shape=(100,))
        self.D = self.create_d(d_input)
        self.G = self.create_g(g_input)
        x = self.G(gan_input)
        gan_output = self.D(x)
        self.gan = Model(inputs=gan_input, outputs=gan_output)

    def create_d(self, d_input):
        #inp = Input(shape=(224, 224, 3))
        x = Conv2D(16, 7, strides=3, activation='elu')(d_input)
        x = Conv2D(16, 7, strides=3, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(32, 5, strides=2, activation='elu')(x)
        x = Conv2D(32, 5, strides=2, activation='elu')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(64, 1, activation='elu')(x)
        x = Conv2D(64, 1, activation='elu')(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=d_input, otuputs=x)
        return model

    def create_g(self, g_input):
        #inp = Input(shape=(100,))
        x = Dense(28*28*12, activation='elu')(g_input)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Reshape((28, 28, 12))(x)
        x = Conv2DTranspose(8, 3, padding='same', activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(6, 3, padding='same', activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(3, 3, padding='same', activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(3, 3, padding='same', activation='sigmoid')(x)
        model = Model(inputs=g_input, outputs=x)
        return model