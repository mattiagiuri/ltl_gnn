import argparse
import copy
import csv
import fcntl
import json
import os

import utils
from utils.logging.json_encoder import JsonEncoder
from utils.logging.logger import Logger


class FileLogger(Logger):
    """
    A simple logger that writes a CSV file.
    """

    def __init__(self, config: argparse.Namespace, resuming: bool = False):
        super().__init__(config)
        self.log_path = utils.get_experiment_path(config)
        self.log_file = f'{self.log_path}/log.csv'
        if resuming:
            if not os.path.exists(self.log_file):
                raise ValueError('Cannot resume logging, log file does not exist!')
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                self.keys = reader.fieldnames
        elif not resuming and os.path.exists(self.log_file):
            raise ValueError('Log file already exists and resuming set to False!')

    def log_config(self):
        config_file = f'{self.log_path}/../experiment_config.json'
        with open(config_file, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:  # metadata file exists already
                f.seek(0)
                previous_config = json.load(f)
                if previous_config != self.config_as_json():
                    raise ValueError('Previous log with different config exists!')
            else:
                config = vars(copy.deepcopy(self.config))
                del config['experiment']
                json.dump(config, f, indent=4, cls=JsonEncoder)
            fcntl.flock(f, fcntl.LOCK_UN)

    def config_as_json(self) -> dict:
        config = vars(copy.deepcopy(self.config))
        del config['experiment']
        return json.loads(json.dumps(config, cls=JsonEncoder))

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        first_log = self.keys is None
        self.check_keys_valid(data)
        with open(self.log_file, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.keys)
            if first_log:
                writer.writeheader()
            writer.writerow(data)

