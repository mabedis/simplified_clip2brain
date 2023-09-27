"""The util functions."""

import os
import configparser


class Directory:
    """Directory jobs."""

    def __init__(self, path: str = None):
        self.path = path

    def get_root_path(self) -> str:
        """Get prject's root directory path.

        Returns:
            str: Project's root directory absolute path.
        """
        index = os.path.dirname(__file__).index('simplified_clip2brain/')
        return os.path.dirname(__file__)[:index+22]

    def check_dir_existence(self) -> bool:
        """Check whether the directory exists or not.
        In case it doesn't exist, the directory will be created.

        Returns:
            bool: True if exists.
        """
        if self.path and not os.path.exists(self.path):
            os.makedirs(self.path)
            return False

        return True


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
        return Directory().get_root_path() + config[self.section][self.entry]
