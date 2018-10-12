import os
from src.processing.datasets import SiameseDatagen, generate_data_set

def process_data(train_path, test_path, method, batch_size, imsize):
    method = method.lower()
    if method == 'siam' or method == 'siamese':
        train_dataset = SiameseDatagen(
            train_path,
            batch_size=batch_size,
            imsize=imsize
        )
        test_dataset = SiameseDatagen(
            test_path,
            batch_size=batch_size,
            imsize=imsize
        )
        return train_dataset, test_dataset
    elif method == 'class' or method == 'classification' or method == 'gan' or method == 'gand':
        train_dataset = generate_data_set(
            train_path,
            augment=True,
            batch_size=batch_size,
            imsize=imsize
        )
        test_dataset = generate_data_set(
            test_path,
            augment=False,
            batch_size=batch_size,
            imsize=imsize
        )
        return train_dataset, test_dataset
    raise ValueError(
        'Provided method {} is not supported'.format(method)
    )