"""Data utils."""

import os
import configparser

import numpy as np
import pandas as pd

from src.utils.model_config import COCO_cat, COCO_super_cat


config = configparser.ConfigParser()
config.read("config.cfg")
stim_path = config["DATA"]["StimuliInfo"]
STIM = pd.read_pickle(stim_path)


def fill_in_nan_voxels(vals, subj: int, output_root: str, fill_in: int = 0):
    """Fill NaN Voxels.

    Args:
        vals (Any): _description_
        subj (int): Subject between 1-8
        output_root (str): Root address
        fill_in (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    try:  # some subject has zeros voxels masked out
        nonzero_mask = np.load(
            f"{output_root}/output/voxels_masks/subj{subj}/nonzero_voxels_subj{subj}.npy"
        )
        if isinstance(vals, list) or len(vals.shape) == 1:
            tmp = np.zeros(nonzero_mask.shape) + fill_in
            tmp[nonzero_mask] = vals
        elif len(vals.shape) == 2:
            tmp = np.zeros((vals.shape[0], len(nonzero_mask))) + fill_in
            tmp[:, nonzero_mask] = vals
        return tmp
    except FileNotFoundError:
        return vals


def load_model_performance(
    model, output_root: str = ".", subj: int = 1, measure: str = "corr"
) -> np.array:
    """Load model.

    Args:
        model (_type_): _description_
        output_root (str, optional): _description_. Defaults to ".".
        subj (int, optional): _description_. Defaults to 1.
        measure (str, optional): _description_. Defaults to "corr".

    Returns:
        np.array: _description_
    """
    if measure == "pvalue":
        measure = "corr"
        pvalue = True
    else:
        pvalue = False

    if isinstance(model, list):
        # to accomodate different naming of the same model
        for name in model:
            try:
                out = np.load(
                    f"{output_root}/output/encoding_results/subj{subj}/{measure}_{name}_whole_brain.p",
                    allow_pickle=True,
                )
            except FileNotFoundError:
                continue
    else:
        out = np.load(
            f"{output_root}/output/encoding_results/subj{subj}/{measure}_{model}_whole_brain.p",
            allow_pickle=True,
        )

    if measure == "corr":
        if pvalue:
            out = np.array(out)[:, 1]
            out = fill_in_nan_voxels(out, subj, output_root, fill_in=1)
            return out
        out = np.array(out)[:, 0]

    out = fill_in_nan_voxels(out, subj, output_root)

    return np.array(out)


def load_top1_objects_in_coco(cid: int):
    """Load Top 1 Objects in COCO dataset.

    Args:
        cid (int): COCO ID.

    Returns:
        _type_: _description_
    """
    cat = np.load("features/cat.npy")

    # extract the nsd ID corresponding to the coco ID in the stimulus list
    stim_ind = STIM["nsdId"][STIM["cocoId"] == cid]
    # extract the respective features for that nsd ID
    cat_id_of_trial = cat[stim_ind, :]

    return COCO_cat[np.argmax(cat_id_of_trial)]


def load_objects_in_coco(cid: int) -> list:
    """Load objects in COCO dataset.

    Args:
        cid (int): COCO ID.

    Returns:
        list: _description_
    """
    cat = np.load("features/cat.npy")
    supcat = np.load("features/supcat.npy")

    # extract the nsd ID corresponding to the coco ID in the stimulus list
    stim_ind = STIM["nsdId"][STIM["cocoId"] == cid]
    # extract the repective features for that nsd ID
    cat_id_of_trial = cat[stim_ind, :].squeeze()
    supcat_id_of_trial = supcat[stim_ind, :].squeeze()
    catnms = []

    assert len(cat_id_of_trial) == len(COCO_cat)
    assert len(supcat_id_of_trial) == len(COCO_super_cat)

    catnms += list(COCO_cat[cat_id_of_trial > 0])
    catnms += list(COCO_super_cat[supcat_id_of_trial > 0])
    return catnms


def load_subset_trials(coco_id_by_trial, cat, negcat: bool = False) -> list:
    """
    Returns a list of idx to apply on the 10,000 trials for each subject.
    These are not trials ID themselves but indexs for trials IDS.
    """
    subset_idx, negsubset_idx = [], []
    for i, _id in enumerate(coco_id_by_trial):
        catnms = load_objects_in_coco(_id)
        if cat in catnms:
            subset_idx.append(i)
        else:
            negsubset_idx.append(i)
    return negsubset_idx if negcat else subset_idx


def find_trial_indexes(
    subj: int, cat: str = "person", output_dir: str = "output"
) -> tuple:
    """Find trial indexes.

    Args:
        subj (int): Subject between 1-8.
        cat (str, optional): Category. Defaults to "person".
        output_dir (str, optional): Output directory. Defaults to "output".

    Returns:
        tuple: indexes.
    """
    coco_id = np.load(f"{output_dir}/coco_ID_of_repeats_subj{subj}.npy")

    idx1, idx2 = [], []
    for i, _id in enumerate(coco_id):
        catnms = load_objects_in_coco(_id)
        if cat in catnms:
            idx1.append(i)
        else:
            idx2.append(i)

    return idx1, idx2


def extract_test_image_ids(subj: int = 1, output_dir: str = "output") -> tuple:
    """Extract test image IDs.

    Args:
        subj (int, optional): Subject between 1-8. Defaults to 1.
        output_dir (str, optional): Output directory. Defaults to "output".

    Returns:
        tuple: Test image's ID and test index.
    """
    from sklearn.model_selection import train_test_split

    _, test_idx = train_test_split(
        range(10000), test_size=0.15, random_state=42)
    coco_id = np.load(f"{output_dir}/coco_ID_of_repeats_subj{subj}.npy")
    test_image_id = coco_id[test_idx]

    return test_image_id, test_idx


def extract_single_roi(roi_name: str, output_dir: str, subj: int) -> tuple:
    """Extract a single RIO.

    Args:
        roi_name (str): ROI's name.
        output_dir (str): Output directory.
        subj (int): Subject between 1-8.

    Returns:
        tuple: output_masks and roi_labels
    """
    from src.utils.model_config import roi_name_dict
    from src.extract_cortical_voxel import ExtractCorticalMask

    output_masks, roi_labels = [], []
    try:
        roi_mask = np.load(
            os.path.join(
                output_dir,
                f"output/voxels_masks/subj{subj}/roi_1d_mask_subj{subj}_{roi_name}.npy",
            )
        )
    except FileNotFoundError:
        cortical_mask = ExtractCorticalMask(
            subjects=subj, roi=roi_name, output_dir=output_dir)
        roi_mask = cortical_mask.extract_corital_mask()
        roi_mask = np.load(
            os.path.join(
                output_dir,
                f"output/voxels_masks/subj{subj}/roi_1d_mask_subj{subj}_{roi_name}.npy",
            )
        )

    roi_dict = roi_name_dict[roi_name]
    for k, v in roi_dict.items():
        if int(k) > 0 and np.sum(roi_mask == int(k)) > 0:
            output_masks.append(roi_mask == int(k))
            roi_labels.append(v)

    return output_masks, roi_labels


def compute_sample_performance(
    model, subj: int, output_dir: str, masking: str = "sig", measure: str = "corrs"
) -> np.array:
    """Returns sample-wise performances for encoding model.

    Args:
        model (_type_): _description_
        subj (int): _description_
        output_dir (str): Output directory to save or load datasets.
        masking (str, optional): Masking. Defaults to "sig".
        measure (str, optional): Type of measurements, e.g: Corrolation or R2.
            Defaults to "corrs".

    Returns:
        np.array: _description_
    """
    if measure == "corrs":
        from scipy.stats import pearsonr
        metric = pearsonr
    elif measure == "rsq":
        from sklearn.metrics import r2_score
        metric = r2_score

    try:
        sample_corrs = np.load(
            f"{output_dir}/output/clip/{model}_sample_{measure}_{masking}.npy"
        )
        if len(sample_corrs.shape) == 2:
            sample_corrs = np.array(sample_corrs)[:, 0]
            np.save(
                f"{output_dir}/output/clip/{model}_sample_corrs_{masking}.npy",
                sample_corrs,
            )
    except FileNotFoundError:
        yhat, ytest = load_model_performance(
            model, output_root=output_dir, measure="pred"
        )
        if masking == "sig":
            pvalues = load_model_performance(
                model, output_root=output_dir, measure="pvalue"
            )
            sig_mask = pvalues <= 0.05

            sample_corrs = [
                metric(ytest[:, sig_mask][i, :], yhat[:, sig_mask][i, :])
                for i in range(ytest.shape[0])
            ]

        else:
            roi = np.load(
                f"{output_dir}/output/voxels_masks/subj{subj}/roi_1d_mask_subj{subj}_{masking}.npy"
            )
            roi_mask = roi > 0
            sample_corrs = [
                metric(ytest[:, roi_mask][i, :], yhat[:, roi_mask][i, :])
                for i in range(ytest.shape[0])
            ]

        if measure == "corr":
            sample_corrs = np.array(sample_corrs)[:, 0]
        np.save(
            f"{output_dir}/output/clip/{model}_sample_{measure}_{masking}.npy",
            sample_corrs,
        )

    return sample_corrs
