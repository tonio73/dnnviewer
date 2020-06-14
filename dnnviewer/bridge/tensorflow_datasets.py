from ..TestData import TestData

from tensorflow import keras

import numpy as np


def keras_load_test_data(dataset_name, test_data: TestData, max_samples=128):
    """ Load dataset using Keras """

    # NOTE: max_samples adjusted to fit with the gradient computation

    test_data.reset()

    _datasets = {'cifar-10': (keras.datasets.cifar10, ['red', 'green', 'blue'],
                              ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
                 'fashion-mnist': (keras.datasets.fashion_mnist, ['bw'],
                                   ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                                    'Bag', 'Ankle boot']),
                 'mnist': (keras.datasets.mnist, ['bw'],
                           [str(d) for d in range(10)]),
                 }

    if dataset_name in _datasets:
        dataset = _datasets[dataset_name]
        (_, _), (x_test, y_test) = dataset[0].load_data()
        test_data.set(x_test[:max_samples], np.squeeze(y_test)[:max_samples], dataset[1], dataset[2])


def keras_test_data_listing():
    """ @return dictionary id: caption of available test datasets """
    return {'cifar-10': "CIFAR 10",
            'fashion-mnist': "Fashion MNIST",
            'mnist': "MNIST digits"}
