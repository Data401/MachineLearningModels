class ConvergenceError(RuntimeError):
    def __init__(self):
        self.message = 'SGD failed to converge while being fit. Lower your learning rate and try again.'
