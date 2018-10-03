import argparse

from src.processing.processing import process_data
from src.training.train import train
from src.models.model import generate_model

parser = argparse.ArgumentParser(description='Semisupervised large image dataset filtering for crowdsourced classification')

parser.add_argument('--method', help='data filtering method, choose one: ALL, siamese, GAN, GANM, baseline',
                    default='ALL')
parser.add_argument('--dataset', help='dataset location',
                    default='data')
parser.add_argument('--arch', help='CNN architecture: resnte, vgg',
                    default='resnet')
parser.add_argument('--savepath', help='path where to save trained model',
                    default='resnet')
parser.add_argument('--modelpath', help='path to trained model',
                    default='')
parser.add_argument('--batch_size', help='training batch size',
                    default=16)
parser.add_argument('--imsize', help='image resize size',
                    default=224)
parser.add_argument('--train', help='requires training',
                    action='store_true', default=False)
parser.add_argument('--epochs', help='training epochs',
                    default=40)

def main():
    args = parser.parse_args()
    method = args.method
    dataset = args.dataset
    arch = args.arch
    batch_size = args.batch_size
    imsize = args.imsize
    if args.train:
        train_gen, test_gen = process_data(
            dataset, 
            method, 
            batch_size, 
            imsize
        )
        model = generate_model(method, arch, args.modelpath)
        if args.modelpath is '':
            model = train(model, train_gen, test_gen,
                            args.epochs, args.savepath)
        
            



if __name__ == '__main__':
    main()