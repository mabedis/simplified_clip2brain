"""Test for Extract Image List."""

import os

import numpy as np
import pytest
from unittest.mock import MagicMock

import sys
sys.path.append('/home/cambyses/codes/NMA/simplified_clip2brain/')
from src.extract_image_list import (
    ExtractImageFactory, ExtractImageCoco, ExtractImageTrial
)


@pytest.fixture
def stimuli():
    stimuli = MagicMock()
    stimuli.cocoId = [1, 2, 3, 4, 5]
    stimuli.subject1_rep0 = [0, 1, 0, 2, 3]
    stimuli.subject1_rep1 = [0, 2, 0, 3, 1]
    stimuli.subject1_rep2 = [0, 3, 0, 1, 2]
    return stimuli


def test_create_extractor_coco(stimuli):
    factory = ExtractImageFactory()
    factory.utils = MagicMock()
    extractor = factory.create_extractor(
        subject=1, image_type='cocoId', output_dir='.')
    assert isinstance(extractor, ExtractImageCoco)


def test_create_extractor_trial(stimuli):
    factory = ExtractImageFactory()
    factory.utils = MagicMock()
    extractor = factory.create_extractor(
        subject=1, image_type='trial', output_dir='.')
    assert isinstance(extractor, ExtractImageTrial)


# def test_extract_image_coco(stimuli):
#     ExtractImageCoco(subject=1, output_dir='.', stimuli=stimuli)
#     col_name = f"subject1_rep{0}"
#     image_id_list = list(stimuli.cocoId[stimuli[col_name] != 0])
#     expected_result = np.array(image_id_list)
#     coco_image_data = np.load(
#         os.path.join('.', "output/coco_ID_of_repeats_subj01.npy")
#     )
#     expected_result = np.array([1, 2, 4, 5])
#     np.testing.assert_array_equal(coco_image_data, expected_result)


# def test_extract_image_trial(stimuli):
#     extractor = ExtractImageTrial(
#         subject=1, output_dir='.', stimuli=stimuli)
#     expected_result = np.array(
#         [[0, 0, 0], [1, 2, 3], [0, 0, 0], [1, 2, 0], [2, 3, 1]])
#     np.testing.assert_array_equal(extractor.result, expected_result)
