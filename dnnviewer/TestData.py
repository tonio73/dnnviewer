

class TestData:
    """ Wrapper for test data """

    x, y = [], []

    has_test_sample = False

    input_classes, output_classes = None, None

    def set(self, x, y, input_classes, output_classes):
        self.x = x
        self.y = y
        self.input_classes = input_classes
        self.output_classes = output_classes
        self.has_test_sample = True

    def reset(self):
        self.x = []
        self.y = []
        self.input_classes = None
        self.output_classes = None
        self.has_test_sample = False
