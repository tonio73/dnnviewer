class ModelError(Exception):
    """ Exception to notify issues while loading models """
    def __init__(self, message):
        self.message = message


class DatasetError(Exception):
    """ Exception to notify issues while loading datasets """
    def __init__(self, ds_name, message):
        self.ds_name = ds_name
        self.message = message
