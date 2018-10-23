import keras
import argparse
import os

from src.processing.processing import process_data
from src.training.train import train
from src.models.model import generate_model
from src.validation.validate import eval_cs_model
from src.processing.datasets import generate_data_set

parser = argparse.ArgumentParser(description='Semisupervised large image dataset filtering for crowdsourced classification')

parser.add_argument('--method', help='data filtering method, choose one: ALL, siamese, GAN, GANM, baseline',
                    default='siamese')
parser.add_argument('--train_path', help='dataset location',
                    default='data/train')
parser.add_argument('--val_path', help='validation data location',
                    default='data/test')
parser.add_argument('--dev_path', help='development data location',
                    default='data/dev')
parser.add_argument('--arch', help='CNN architecture: resnte, vgg',
                    default='resnet')
parser.add_argument('--savepath', help='path where to save trained model',
                    default='resnet')
parser.add_argument('--modelpath', help='path to trained model',
                    default='')
parser.add_argument('--batch_size', help='training batch size',
                    default=16, type=int)
parser.add_argument('--imsize', help='image resize size',
                    default=224)
parser.add_argument('--train', help='requires training',
                    action='store_true', default=False)
parser.add_argument('--epochs', help='training epochs',
                    default=150, type=int)
parser.add_argument('--gpu', help='gpu device to use',
                    default="0")

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_gen, test_gen = process_data(
            args.train_path,
            args.val_path,
            args.method,
            args.batch_size,
            args.imsize
    )
    model = generate_model(args.method, args.arch, args.modelpath)

    if args.train and args.modelpath is '':
        model = train(args.method, model, train_gen, test_gen,
                        args.epochs, args.savepath)
    elif args.modelpath is not '':
        model = keras.models.load_model(args.modelpath)
    else:
        raise ValueError('Without training, path to pretrained model ' +
                        'not provided')
    dev_gen = generate_data_set(args.dev_data, False, args.batch_size,
                                args.imsize)
    eval_cs_model(model, dev_gen, test_gen)

if __name__ == '__main__':
    main()