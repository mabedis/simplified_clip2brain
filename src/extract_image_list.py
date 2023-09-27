"""Extract Image COCO_ID and Image Trial."""

import os

import numpy as np
import pandas as pd

from src.utils import utils


class ExtractImageFactory:
    """Image extraction using Factory design pattern."""

    def create_extractor(self, subject: int, image_type: str, output_dir: str):
        """Create image extractor.

        Args:
            subject (int): Subject in range 1-8
            image_type (str): cocoId or trial
            output_dir (str): Output directory path

        Raises:
            Exception: Invalid image type.

        Returns:
            numpy.ndarray: _description_
        """
        stimuli_path = utils.GetNSD(section='DATA', entry='StimuliInfo')
        stimuli = pd.read_pickle(stimuli_path.get_dataset_path())
        directory = utils.Directory(path=os.path.join(output_dir, 'output'))
        directory.check_dir_existence()

        if image_type == 'cocoId':
            return ExtractImageCoco(subject=subject, output_dir=output_dir, stimuli=stimuli)
        elif image_type == 'trial':
            return ExtractImageTrial(subject=subject, output_dir=output_dir, stimuli=stimuli)
        else:
            raise ValueError(f'Invalid image type: {image_type}.')


class ExtractImageCoco:
    """Extract image from COCO dataset."""

    def __init__(self, subject: int, output_dir: str, stimuli):
        # Using the first rep b/c three repetitions should have the same order of IDs.
        col_name = f"subject{subject}_rep{0}"
        image_id_list = list(
            stimuli.cocoId[stimuli[col_name] != 0])
        self.result = np.array(image_id_list)
        np.save(
            os.path.join(
                output_dir, f'output/coco_ID_of_repeats_subj{subject:02}.npy'),
            self.result,
        )


class ExtractImageTrial:
    """Extract image from trial."""

    def __init__(self, subject: int, output_dir: str, stimuli):
        all_rep_trials_list = [list(
            stimuli[f"subject{subject}_rep{rep}"][stimuli[f"subject{subject}_rep{rep}"] != 0])
            for rep in range(3)]
        self.result = np.array(all_rep_trials_list).T - 1
        np.save(
            os.path.join(
                output_dir, f'output/trials_subj{subject:02}.npy'),
            self.result,
        )
