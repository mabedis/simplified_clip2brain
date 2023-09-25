"""The util functions."""

import os
import configparser


def get_root_path() -> str:
    """Get prject's root directory path.

    Returns:
        str: Project's root directory absolute path.
    """
    index = os.path.dirname(__file__).index('simplified_clip2brain/')
    return os.path.dirname(__file__)[:index+22]


class GetNSD:
    """Get NSD directories."""

    def __init__(self, section: str, entry: str):
        self.section = section
        self.entry = entry

    def get_dataset_path(self) -> str:
        """Get the path to NSD datasets.

        Returns:
            str: Absolute path to NSD datasets.
        """
        config = configparser.ConfigParser()
        config.read("config.cfg")
        return get_root_path() + config[self.section][self.entry]
