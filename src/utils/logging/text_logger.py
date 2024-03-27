from train.experiment_metadata import ExperimentMetadata
from utils.logging.logger import Logger


class TextLogger(Logger):
    """
    A logger that logs to standard output.
    """

    def __init__(self, experiment: ExperimentMetadata):
        super().__init__(experiment)
        self.log_metadata()

    def log_metadata(self):
        print(self.metadata)

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        self.check_keys_valid(data)
        row = ''
        for key, value in data.items():
            short_name = self.get_short_name(key)
            row += f'{short_name}: '
            if isinstance(value, float):
                row += f'{value:.2f} | '
            else:
                row += f'{value} | '
        row = row[:-3]  # remove trailing ' | '
        print(row)

    @staticmethod
    def get_short_name(key: str) -> str:
        if key == 'return_per_episode_mean':
            return 'rμ'
        elif key == 'return_per_episode_std':
            return 'rσ'
        elif key == 'num_frames_per_episode_mean':
            return 'fμ'
        elif key == 'num_frames_per_episode_std':
            return 'fσ'
        elif key == 'duration':
            return 't'
        else:
            return key

    @staticmethod
    def info(message: str):
        print(message)
