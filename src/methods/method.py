from src.processing.processing import process_data

from src.methods.gan import process_gan
from src.methods.baseline import process_baseline

from functools import reduce

def process_method(method):
    method = method.lower()
    if method is 'siam':
        return process_baseline
    if method is 'gan':
        return process_gan
    if method is 'baseline':
        return process_baseline
    if method is 'gand':
        return process_gan
    if method is 'ae':
        return process_baseline
    raise ValueError('Specified method {} is invalid'.format(method))