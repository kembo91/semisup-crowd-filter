from src.processing.processing import process_data

from src.methods.siamese import process_siamese
from src.methods.gan import process_gan
from src.methods.ae import process_ae
from src.methods.gand import process_gand
from src.methods.baseline import process_baseline

from functools import reduce

def process_methods(methods):
    print(methods)
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
            reduce(methods, lambda l,r: l + ', ' + r))
        )
    return meth

def pipeline(dataargs, method):
    outputs = []
    datagen = process_data(**dataargs)
    outputs.append(method(data))
