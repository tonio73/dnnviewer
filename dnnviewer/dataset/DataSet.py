from .generators import AbstractGenerator


class DataSet:
    """ Wrapper for test data """

    MODE_UNKNOWN = 0
    MODE_FILESET = 1
    MODE_GENERATOR = 2

    def __init__(self):
        self.mode = DataSet.MODE_UNKNOWN

        self.x = []
        # Formatted for input of the model (using first model as reference in case of sequence)
        self.x_format = []
        self.y = []

        self.input_classes = None
        self.output_classes = None
        # Reference to the generator object when using
        self.generator: AbstractGenerator = None

    def set(self, x, y, input_classes, output_classes):
        self.x = x
        self.y = y

        self.input_classes = input_classes
        self.output_classes = output_classes

        self.mode = DataSet.MODE_FILESET

    def reset(self):
        self.mode = DataSet.MODE_UNKNOWN

        self.x = []
        self.x_format = []
        self.y = []

        self.input_classes = None
        self.output_classes = None

        self.generator = None
