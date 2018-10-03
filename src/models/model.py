import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Flatten, Reshape, Conv2DTranspose, GlobalMaxPool2D

from src.utils.utils import euclidean_distance
from src.utils.losses import contrastive_loss



def generate_model(method, type, path):
    if path is not '':
        model = load_model(path)
        return model
    elif method is 'siam' or method is 'siamese':
        return create_siam_model(type)
    elif method is 'class' or method is 'classification':
        return create_class_model(type)
    raise ValueError(
        'Specified method {} is not supported'.format(method)
    )
        
def get_pretrained_model(type):
    type = type.lower()
    if 'resnet' in type:
        return keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet'
        )
    elif 'vgg' in type:
        return keras.applications.vgg16.VGG16(
            include_top=False,
            weights='imagenet'
        )
    raise ValueError('Network type {} not recognized'.format(type))

def create_siam_model(type):
    input_img = Input(shape=(224, 224, 3))
    net = get_pretrained_model(type)
    x = net(input_img)
    output = GlobalMaxPool2D()(x)
    output = Lambda(lambda x: K.l2_normalize(x, axis=1))(output)
    resnet_model = Model(input_img, output)
    input1 = Input(shape = (224, 224, 3))
    input2 = Input(shape = (224, 224, 3))
    o1 = resnet_model(input1)
    o2 = resnet_model(input2)
    distance = Lambda(euclidean_distance)([o1, o2])
    model = Model(inputs = [input1, input2], outputs = distance)
    adam = keras.optimizers.adam(lr = 1e-4)
    model.compile(optimizer=adam, loss = contrastive_loss)
    return model

def create_class_model(type, num_classes=8):
    input_img = Input(shape=(224, 224, 3))
    net = get_pretrained_model(type)
    x = net(input_img)
    x = Dense(2048, activation='elu')(x)
    x = Dense(1024, activation='elu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=x)
    adam = keras.optimizers.adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    return model
    
def create_gan_model(type):
    print('TODO')