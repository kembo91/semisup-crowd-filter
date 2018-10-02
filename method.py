from . import processing

from methods.siamese import process_siamese
from methods.gan import process_gan
from methods.ae import process_ae
from methods.gand import process_gand
from methods.baseline import process_baseline

from functools import reduce

def process_methods(methods):
    meth = []
    for method in methods:
        method = method.lower()
        if method is 'siam':
            meth.append(process_siamese)
        if method is 'gan':
            meth.append(process_gan)
        if method is 'baseline':
            meth.append(process_baseline)
        if method is 'gand':
            meth.append(process_gand)
        if method is 'ae':
            meth.append(process_ae)
    if len(meth) == 0:
        print('listed methods are invalid {}'.format(
            reduce(lambda l,r: l + ' ' + r), methods)
        )
    return meth

def pipeline(datapaths, methods):
    outputs = []
    for path in datapaths:
        data = process_data(path)
        for method in methods:
            outputs.append(method(data))

            
    