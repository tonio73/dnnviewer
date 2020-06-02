

class Progress:
    """ Manage progress step, status and information message """

    # Statuses
    NONE = 0
    INFO = 1
    WARN = 2
    ERROR = 3

    def __init__(self, num_steps: int = None):
        self.num_steps = num_steps
        self.statuses = []
        self.current_step = 0
        self.next = ''

    def reset(self, num_steps=None):
        self.statuses = []
        self.current_step = 0
        if num_steps is not None:
            self.num_steps = num_steps

    def set_next(self, next_):
        self.next = next_

    def forward(self, step_delta=1, status=NONE, status_msg=""):
        self.current_step += step_delta
        self.next = ''
        self.statuses.append((self.current_step, status, status_msg))

    def get_status(self):
        if self.statuses:
            return self.statuses[-1]
        return 0, Progress.NONE, ""
