import os

from random import shuffle, choice
import numpy as np
from PIL import Image
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict

class SiameseDatagen(Sequence):
    def __init__(self, datadir, batch_size = 16, imsize = 224, augment=True):
        self.datadir = datadir
        self.listims = []
        self.bs = batch_size
        self.imsize = imsize
        self.imsdict = defaultdict(list)
        self.listims = []
        self.augment = augment
        for root, _, files in os.walk(datadir):
            if len(files) is not 0:
                for file in files:
                    path = os.path.join(root, file)
                    splitpath = path.split(os.sep)
                    self.imsdict[splitpath[2]].append(path)
                    self.listims.append(path)
        self.indexes = list(range(len(self.listims)))
    
    def __len__(self):
        counter = 0
        for key in self.imsdict:
            counter += len(self.imsdict[key])
        return counter // self.bs
    
    def choose_image(self, cls, negative = True):
        if negative:
            choose_lst = [key for key in self.imsdict if key is not cls] 
        else:
            choose_lst = [cls]
        cls = np.random.choice(choose_lst, size = 1)[0]
        rndim = np.random.choice(self.imsdict[cls], size = 1)[0]
        return rndim
    
    def load_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.imsize, self.imsize))
        img = np.asarray(img) / 255
        img = img[np.newaxis, ...]
        return img
    
    def generate_batch(self, inds):
        batch_anc = np.ndarray(shape = (0, self.imsize, self.imsize, 3))
        batch_img = np.ndarray(shape = (0, self.imsize, self.imsize, 3))
        label = np.array([])
        for ind in inds:
            impath = self.listims[ind]
            anc_im = self.load_image(impath)
            cls = impath.split(os.sep)[2]
            negim = self.choose_image(cls, True)
            posim = self.choose_image(cls, False)
            neg_im = self.load_image(negim)
            pos_im = self.load_image(posim)
            
            batch_anc = np.vstack((batch_anc, anc_im))
            batch_img = np.vstack((batch_img, pos_im))
            batch_anc = np.vstack((batch_anc, anc_im))
            batch_img = np.vstack((batch_img, neg_im))
            label = np.append(label, [1, 0])
        return [batch_anc, batch_img], label
    
    def choose(self, elem):
        i = choice(self.indexes)
        if i == elem:
            return self.choose(elem)
        return i
    
    def on_epoch_end(self):
        shuffle(self.indexes)
    
    def __getitem__(self, idx):
        indicies = self.indexes[self.bs//2 * idx : self.bs//2 * (idx + 1)]
        return self.generate_batch(indicies)

def generate_data_set(datapath, augment, batch_size=16, imsize=224):
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1./255
        )
    generator = datagen.flow_from_directory(
        datapath,
        batch_size=batch_size,
        target_size=imsize,
        class_mode='categorical'
    )
    return generator