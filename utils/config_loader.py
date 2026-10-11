'''
    Loading the configuration from the yaml file
'''
import yaml


class Config:
    """Turns a (possibly nested) dict into an object with attribute access."""

    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"


# loading the configuration file
class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path

    def load(self):
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
        return Config(data)
