import os
import keras
import keras.backend as K
from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, Conv2DTranspose, GlobalMaxPool2D
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from src.utils.utils import schedule
from tqdm import tqdm
import numpy as np

def train(method, model, train_set, val_set, epochs, savepath):
    print('training {} model'.format(method))
    if method != 'gan':
        return train_classic(model, train_set, val_set, epochs, savepath)
    else:
        return train_gan(model, train_set, val_set, epochs, savepath)

def train_classic(model, train_set, val_set, epochs, savepath):
    path = 'checkpoints/{}'.format(savepath)
    if not os.path.exists(path):
        os.makedirs(path)
    checkpointer = ModelCheckpoint(
        path, monitor='val_loss', save_best_only=True
    )
    scheduler = LearningRateScheduler(schedule, verbose = 1)
    model.fit_generator(
        train_set,
        validation_data=val_set,
        steps_per_epoch=200,
        epochs=epochs,
        callbacks=[checkpointer]
    )
    model = keras.models.load_model(path)
    return model

def train_gan(model, train_set, val_set, epochs, savepath):
    path = 'checkpoints/{}'.format(savepath)
    if not os.path.exists(path):
        os.makedirs(path)
    checkpointer = ModelCheckpoint(
        path, monitor='val_loss', save_best_only=True
    )
    scheduler = LearningRateScheduler(schedule, verbose = 1)
    losses = {"d":[], "g":[]}
    tmpDloss = []
    tmpGloss = []
    #train discriminator
    print('Training discriminator')
    for epoch in tqdm(range(epochs)):
        for ix in range(len(train_set)):
            image_batch = train_set[ix][0]
            batch_size = image_batch.shape[0]
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            generated_batch = model.G.predict(noise)

            X = np.concatenate((image_batch, generated_batch))
            y = np.zeros([2*batch_size, 2])
            y[0:batch_size, 0] = 1
            y[batch_size:, 1] = 1

            d_loss = model.D.train_on_batch(X, y)
            losses['d'].append(d_loss)
            tmpDloss.append(d_loss)

            noise = np.random.normal(0, 1, size=[2*batch_size, 100])
            y_mis = np.zeros([2*batch_size, 2])
            y_mis[0:2*batch_size, 1] = 1
            g_loss = model.GAN.train_on_batch(noise, y_mis)

            losses['g'].append(g_loss)
            tmpGloss.append(g_loss)
        print('epoch: #{}, D loss:{}'.format(epoch, np.mean(tmpDloss)))
        print('epoch: #{}, G loss:{}'.format(epoch, np.mean(tmpGloss)))
        tmpDloss = []
        tmpGloss = []
        model.GAN.save('checkpoints/gan/ganmodel_{}'.format(epoch))
        