from . import DatasetError
from ..TestData import TestData

import tensorflow_datasets as tf_ds

import logging

_logger = logging.getLogger(__name__)


def load_test_data(dataset_name, test_data: TestData, max_samples=128):
    """ Download and prepare a Tensorflow dataset to be used as test data """
    try:
        ds, info = tf_ds.load(dataset_name,
                              split=None,
                              batch_size=max_samples,
                              as_supervised=True,
                              with_info=True)
        # Select 'test' split of any if missing
        if 'test' in ds.keys():
            ds_test = ds['test']
        else:
            ds_test = list(ds.values())[0]

        # Get first batch
        ds_np = next(tf_ds.as_numpy(ds_test))

        # Tensorflow Dataset does not provide the labels as is => use own listing
        in_labels, out_labels = None, None
        if dataset_name in _class_labels.keys():
            labels = _class_labels[dataset_name]
            in_labels = labels['in']
            out_labels = labels['out']

        test_data.set(ds_np[0], ds_np[1], in_labels, out_labels)  # As supervised mode => X / y

    except Exception as e:
        _logger.debug(str(e))
        raise DatasetError(dataset_name, "Error while loading dataset: %s" % str(e))


def list_test_data():
    """ List all datasets in Tensorflow Dataset """
    return tf_ds.list_builders()


_class_labels = {
    'cifar10': {
        'in': ['red', 'green', 'blue'],
        'out': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    },
    'fashion_mnist': {
        'in': ['bw'],
        'out': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot']
    }
}
