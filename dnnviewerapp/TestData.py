

class TestData:
    """ Wrapper for test data """

    test_sample_init = 0

    x_test, y_test = [], []

    has_test_sample = False

    input_classes, output_classes = None, None

    def set(self, x, y, input_classes, output_classes):
        self.x_test = x
        self.y_test = y
        self.input_classes = input_classes,
        self.output_classes = output_classes
        self.has_test_sample = True
