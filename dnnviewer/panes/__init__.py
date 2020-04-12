

class AbstractPane:
    """ Abstract pane """

    def render(self, grapher):
        """ Render before extracting the layout """
        return

    def get_layout(self, model_sequence, grapher, test_data):
        """ Get pane layout """
        return

    def callbacks(self, dash_app, model_sequence, grapher, test_data):
        """ Local callbacks """
        return
