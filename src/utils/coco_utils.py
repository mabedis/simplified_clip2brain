"""Helper function for COCO dataset."""

import numpy as np
from skimage import io
from pycocotools.coco import COCO

from src.utils import utils


class COCOUtils:
    """Utility class for COCO datasets."""

    def __init__(self):
        train_caps_path = utils.GetNSD(
            section='COCO', entry='AnnFileTrainCaps')
        ann_file_train_caps = train_caps_path.get_dataset_path()
        val_caps_path = utils.GetNSD(section='COCO', entry='AnnFileValCaps')
        ann_file_val_caps = val_caps_path.get_dataset_path()

        self.train_caps = COCO(ann_file_train_caps)
        self.val_caps = COCO(ann_file_val_caps)

    def load_captions(self, cid: int) -> list:
        """Load captions from COCO dataset.

        Args:
            cid (int): Caption ID.

        Returns:
            list: List of captions.
        """
        annIds = self.train_caps.getAnnIds(imgIds=[cid])
        anns: list = self.train_caps.loadAnns(annIds)

        if not any(anns):
            annIds = self.val_caps.getAnnIds(imgIds=[cid])
            anns = self.val_caps.loadAnns(annIds)

        if not any(anns):
            print("no captions extracted for image: " + str(cid))

        return [d["caption"] for d in anns]

    def get_coco_image(self, _id: int, coco_train, coco_val) -> np.ndarray:
        """Get an image from COCO dataset.

        Args:
            _id (int): _description_
            coco_train (_type_): _description_
            coco_val (_type_): _description_

        Returns:
            np.ndarray: The read image is returned as a NumPy array
        """
        try:
            img = coco_train.loadImgs([_id])[0]
        except KeyError:
            img = coco_val.loadImgs([_id])[0]

        return io.imread(img["coco_url"])

    def get_coco_anns(self, _id: int, coco_train, coco_val) -> list:
        """Get annotations from COCO dataset.

        Args:
            _id (int): _description_
            coco_train (_type_): _description_
            coco_val (_type_): _description_

        Returns:
            list: List of categories.
        """
        try:
            annIds = coco_train.getAnnIds([_id])
            anns = coco_train.loadAnns(annIds)
        except KeyError:
            annIds = coco_val.getAnnIds([_id])
            anns = coco_val.loadAnns(annIds)

        return [ann["category_id"] for ann in anns]

    def get_coco_caps(self, _id: int, coco_train_caps, coco_val_caps) -> list:
        """Get captions from COCO dataset.

        Args:
            _id (int): _description_
            coco_train_caps (_type_): _description_
            coco_val_caps (_type_): _description_

        Returns:
            list: List of captions.
        """
        try:
            annIds = coco_train_caps.getAnnIds([_id])
            anns = coco_train_caps.loadAnns(annIds)
        except KeyError:
            annIds = coco_val_caps.getAnnIds([_id])
            anns = coco_val_caps.loadAnns(annIds)

        return [ann["caption"] for ann in anns]
