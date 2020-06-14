class ModelError(Exception):
    """ Exception to notify issues in the bridge package """
    def __init__(self, message):
        self.message = message