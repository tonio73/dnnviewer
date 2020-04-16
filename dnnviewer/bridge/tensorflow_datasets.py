from ..TestData import TestData

from tensorflow import keras


def keras_load_test_data(dataset_name, test_data: TestData):
    """ Load dataset using Keras, return a sample of the test """

    test_data.reset()

    _datasets = {'cifar-10': (keras.datasets.cifar10, ['red', 'green', 'blue'],
                              ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
                 'mnist': (keras.datasets.mnist, ['bw'],
                           [str(d) for d in range(10)]),
                 'fashion-mnist': (keras.datasets.fashion_mnist, ['bw'],
                                   ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                                    'Bag', 'Ankle boot'])
                 }

    if dataset_name in _datasets:
        dataset = _datasets[dataset_name]
        (_, _), (x_test, y_test) = dataset[0].load_data()
        test_data.set(x_test, y_test, dataset[1], dataset[2])


def keras_test_data_listing():
    """ @return dictionary id: caption of available test datasets """
    return {'mnist': "MNIST digits",
            'fashion-mnist': "Fashion MNIST",
            'cifar-10': "CIFAR 10"}
