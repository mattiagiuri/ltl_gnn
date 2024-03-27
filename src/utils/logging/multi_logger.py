from utils.logging.logger import Logger


class MultiLogger(Logger):
    """
    A logger that logs to multiple loggers.
    """

    def __init__(self, *loggers: Logger):
        metadata = loggers[0].metadata if len(loggers) > 0 else {}
        super().__init__(metadata)
        for logger in loggers:
            if logger.metadata != metadata:
                raise ValueError("All loggers must have the same metadata.")
        self.loggers = loggers
        self.log_metadata()

    def log_metadata(self):
        for logger in self.loggers:
            logger.log_metadata()

    def log(self, data: dict[str, float | list[float]]):
        for logger in self.loggers:
            logger.log(data)

    def save_training_status(self, status: dict[str, any]):
        for logger in self.loggers:
            logger.save_training_status(status)

    def save_best_model(self, status: dict[str, any]):
        for logger in self.loggers:
            logger.save_best_model(status)

    def finish(self):
        for logger in self.loggers:
            logger.finish()
