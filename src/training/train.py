import os
import keras
import keras.backend as K
from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, Conv2DTranspose, GlobalMaxPool2D
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

def train(model, train_set, val_set, epochs, savepath):
    path = 'checkpoints/{}'.format(savepath)
    if not os.path.exists(path):
        os.makedirs(path)
    checkpointer = ModelCheckpoint(
        path, monitor='val_loss', save_best_only=True
    )
    print(train_set[0])
    model.fit_generator(
        train_set,
        validation_data=val_set,
        steps_per_epoch=200,
        epochs=epochs,
        callbacks=[checkpointer]
    )
    model = keras.load_model(path)
    return model