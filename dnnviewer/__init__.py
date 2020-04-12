
class AbstractDashboard:
    """ Pure abstract template for the page dashboards """

    def layout(self, has_request: bool):
        """ @return layout """
        return []

    def callbacks(self):
        """ Setup callbacks """
        return
