import csv
import fcntl
import json
import os

import utils
from train.experiment_metadata import ExperimentMetadata
from utils.logging.json_encoder import JsonEncoder
from utils.logging.logger import Logger


class FileLogger(Logger):
    """
    A simple logger that writes a CSV file.
    """

    def __init__(self, experiment: ExperimentMetadata, resuming: bool = False):
        super().__init__(experiment)
        self.log_path = utils.get_experiment_path(experiment)
        self.log_file = f'{self.log_path}/log.csv'
        if resuming:
            if not os.path.exists(self.log_file):
                raise ValueError('Cannot resume logging, log file does not exist!')
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                self.keys = reader.fieldnames
        elif not resuming and os.path.exists(self.log_file):
            raise ValueError('Log file already exists and resuming set to False!')
        self.log_metadata()

    def log_metadata(self):
        metadata_file = f'{self.log_path}/../experiment_metadata.json'
        with open(metadata_file, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:  # metadata file exists already
                f.seek(0)
                previous_config = json.load(f)
                if previous_config != self.metadata_as_json()['config']:
                    raise ValueError('Previous log with different config exists!')
            else:
                json.dump(self.metadata.config, f, indent=4, cls=JsonEncoder)
            fcntl.flock(f, fcntl.LOCK_UN)

    def metadata_as_json(self) -> dict:
        return json.loads(json.dumps(self.metadata, cls=JsonEncoder))

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        first_log = self.keys is None
        self.check_keys_valid(data)
        with open(self.log_file, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.keys)
            if first_log:
                writer.writeheader()
            writer.writerow(data)

