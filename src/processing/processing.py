import os
from src.processing.datasets import SiameseDatagen, ClfDatagen

def process_data(path, method, batch_size, imsize, augment=True):
    method = method.lower()
    if method is 'siam' or method is 'siamese':
        train_dataset = SiameseDatagen(
            os.path.join(path, 'train'),
            batch_size=batch_size,
            imsize=imsize
        )
        test_dataset = SiameseDatagen(
            os.path.join(path, 'test'),
            batch_size=batch_size,
            imsize=imsize
        )
        return train_dataset, test_dataset
    elif method is 'class' or method is 'classification':
        train_dataset = ClfDatagen(
            os.path.join(path, 'train'),
            augment=augment,
            batch_size=batch_size,
            imsize=imsize
        )
        test_dataset = ClfDatagen(
            os.path.join(path, 'test'),
            augment=augment,
            batch_size=batch_size,
            imsize=imsize
        )
        return train_dataset, test_dataset
    raise ValueError(
        'Provided method {} is not supported'.format(method)
    )
        
    