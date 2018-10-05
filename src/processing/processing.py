import os
from src.processing.datasets import SiameseDatagen, generate_data_set

def process_data(path, method, batch_size, imsize):
    method = method.lower()
    if method == 'siam' or method == 'siamese':
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
        train_dataset = generate_data_set(
            os.path.join(path, 'train'),
            augment=True,
            batch_size=batch_size,
            imsize=imsize
        )
        test_dataset = generate_data_set(
            os.path.join(path, 'test'),
            augment=False,
            batch_size=batch_size,
            imsize=imsize
        )
        return train_dataset, test_dataset
    else:
        raise ValueError(
            'Provided method {} is not supported'.format(method)
        )