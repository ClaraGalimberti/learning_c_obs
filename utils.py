import os
import logging
import json
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class WrapLogger:
    def __init__(self, verbose=True):
        now = datetime.now().strftime("%m_%d_%H_%M_%S")
        save_path = os.path.join(ROOT_DIR, "experiments", "logs")
        self.folder = os.path.join(save_path, "logs" + "_" + now)
        os.makedirs(self.folder)
        logging.basicConfig(
            filename=os.path.join(self.folder, "log.log"),
            format="%(asctime)s %(message)s",
            filemode="w",
        )
        logger = logging.getLogger("component" + "_")
        logger.setLevel(logging.DEBUG)
        self.can_log = logger is not None
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)


class Params:
    """Class that contains all the hyperparameters of the running procedure
    Optionally, one can load some parameters through a json file
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """
    def __init__(self, json_path=None):
        if json_path:
            with open(json_path) as f:
                params = json.load(f)
        else:
            params = {}
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    def text_to_print(self):
        msg = "Parameters:\n"
        max_key_length = max(len(key) for key in self.__dict__)
        for key, value in self.__dict__.items():
            msg += f"  {key.ljust(max_key_length)} : {value} \n"
        return msg
