# Developer guide

## Developer install

Clone the repository on your local machine:

```shell script
$ git clone https://github.com/tonio73/dnnviewer.git
```

## Run

For local development, the entry point is the file `dnnviewer.py` in the root directory.

Model ZOO is in 

```shell script
$ dnnviewer.py --model-keras dnnviewer-data/models/MNIST_dense128.h5 --test-dataset mnist
```

# Software architecture

The application is based on Dash for the user interface management (which is generating ReactJs components), and Plotly for the data visualization.

Code is separated in two branches:

- The Graphical representation
- Adapters to read existing models, currently only supporting Keras

## Graphical representation

Top level class it the dnnviewer.Grapher. It holds a collection of layers based on the base class dnnviewer.layers.AbstractLayer.AbstractLayer

## _Bridge_ - adapters to load neural network representation

Within `dnnviewer.bridge` is the code to load a DNN model in Keras and extract the DNN topology, weights, compute the gradients and activations

# Launching the application while developping

## Preselect model and data

#### CIFAR-10 Convolutional neural network (at the beginning of training, single model)

```shell
$ dnnviewer --model-keras dnnviewer-data/models/CIFAR-10_CNN5-Reg.tf --test-dataset cifar10
```

#### MNIST Convolutional neural network based on LeNet5 (single model)

```shell
$ dnnviewer --model-keras dnnviewer-data/models/MNIST_LeNet60.h5 --test-dataset mnist
```

#### Fashion MNIST convolutionnal network (sequence of models)

```shell
$ dnnviewer --sequence-keras "dnnviewer-data/models/FashionMNIST_checkpoints/model1_{epoch}" --test-dataset fashion-mnist
```

## Debug options

`--debug` is launching the Dash application in debug mode:

- The application is automatically reloaded when the code is saved
- The Dash debug tools are shown on screen (errors, callback dep graph)

`--log-level DEBUG`  to get the debug messages displayed in the console log 

# Tooling

### Code quality

- Code is PEP8 compliant, thanks to *flake8* and *Intellij* for watching
- There are a few pytest unit tests in `unittests/`

### Logging and exceptions

Logging is done using the [standard Python logs](https://docs.python.org/2/library/logging.html)

```python
import logging

#...
def my_smart_function():
    logger = logging.getLogger(__name__)

    logger.error('Not smart enough')
```

Exceptions shall be used whenever the  code execution shall be interrupted

##### Exceptions within the _bridge_ package

Exceptions within the _bridge_ package are based on the exception class `bridge.AbstractModelSequence.ModelError`

##### Exceptions within Dash callbacks

 Dash callbacks are safeguarded, no need to add the `try-except` around the callback code.

 To prevent update, the exception to raise is `dash.exceptions.PreventUpdate`

### Run unit test

From project root directory:

```shell script
$ python -m pytest
```

or, simpler:
```shell script
$ make tests
```

#### Code linting

```shell script
$ make lint
```

### Create and upload package

First increase the version number in `setup.py`

Then:

```shell script
$ make package
$ make package_upload
```

Pypi.org credentials required

To force reinstall of the package (in case of minor version increase), do not forget the _--no-deps_:
```shell script
$ pip install --force-reinstall --no-deps dnnviewer 
```