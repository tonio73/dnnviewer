

class Progress:

    NONE=0
    INFO=1
    WARN=2
    ERROR=3

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.statuses = []
        self.current_step = 0

    def reset(self):
        self.statuses = []
        self.current_step = 0

    def progress(self, step_delta=1, status=NONE, status_msg=""):
        self.current_step += step_delta
        self.statuses.append((self.current_step, status, status_msg))

    def get_status(self):
        if self.statuses:
            return self.statuses[-1]
        return 0, Progress.NONE, ""
