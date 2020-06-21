from dnnviewer.bridge.KerasModelSequence import KerasModelSequence
from dnnviewer.dataset.DataSet import DataSet


def test_keras_list_models():

    model_sequence = KerasModelSequence(DataSet())

    model_paths = model_sequence.list_models(['./dnnviewer-data/models'])

    assert len(model_paths) > 0


def test_keras_list_model_sequences():

    model_sequence = KerasModelSequence(DataSet())

    model_paths = model_sequence.list_models(['./dnnviewer-data/models/FashionMNIST_checkpoints'], '{model}_{epoch}')

    assert len(model_paths) == 3
    assert model_paths[0] == 'dnnviewer-data/models/FashionMNIST_checkpoints/model1_{epoch}'
    assert model_paths[1] == 'dnnviewer-data/models/FashionMNIST_checkpoints/model2_{epoch}'
    assert model_paths[2] == 'dnnviewer-data/models/FashionMNIST_checkpoints/model3_{epoch}'
