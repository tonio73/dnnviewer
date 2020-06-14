

class TestData:
    """ Wrapper for test data """

    def __init__(self):
        self.has_test_sample = False

        self.x = []
        # Formatted for input of the model (using first model as reference in case of sequence)
        self.x_format = []
        self.y = []

        self.input_classes = None
        self.output_classes = None

    def set(self, x, y, input_classes, output_classes):
        self.x = x
        self.y = y

        self.input_classes = input_classes
        self.output_classes = output_classes

        self.has_test_sample = True

    def reset(self):
        self.has_test_sample = False

        self.x = []
        self.x_format = []
        self.y = []

        self.input_classes = None
        self.output_classes = None
