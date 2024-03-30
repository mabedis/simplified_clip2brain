# Simplified CLIP2Brain Documentation

## About

## Installation

Note that to install CLIP using pip, try to install it directly from the OpenAI CLIP repo.
The command `pip install clip` may install a different module that has nothing to do with Contrastive Language-Image Pre-Training (CLIP).

## How to use

By default, no parameter is needed. In this case, the following action and parameters are selected by default.

> `python3 main.py --action extract_image_list --subj 1 --type trial`

1. Extract Image List:

    The two commands below are equivalent:
    * > `python3 main.py`
    * > `python3 main.py --action extract_image_list --subject 1 --type trial`

    Other valid commands:
    * > `python3 main.py --action extract_image_list`
    * > `python3 main.py --action extract_image_list --subject 1 --type cocoId`
    * > `python3 main.py --action extract_image_list --subject 1 --type trial`

2. Extract Cortical Voxel:

    * > `python3 main.py --action extract_cortical_voxel`
    * > `python3 main.py --action extract_cortical_voxel --roi floc-bodies`
    * > `python3 main.py --action extract_cortical_voxel --roi floc-bodies --zscore_by_run`
    * > `python3 main.py --action extract_cortical_voxel --mask_only --roi floc-bodies`

3. Compute EV:

    * Compute explainable variance for the data and output data averaged by repeats
    * > `python main.py --action compute_ev --subject 1 --zscored_input --compute_ev`
    * > `python main.py --action compute_ev --subject 1 --compute_ev`
    * > `python main.py --action compute_ev --subject 1 --zscored_input --compute_ev --compute_sample_ev`
    * > `python main.py --action compute_ev --subject 1 --compute_ev --compute_sample_ev`

5. Extract CLIP Features:

    * > `python main.py --action extract_clip_features`
    * > `python main.py --action extract_clip_features [--subject 1] | [--output_dir <path to output_dir>] | [--feature_dir <path to feature_dir>]`

## Sources
- [![DOI](https://zenodo.org/badge/663684836.svg)](https://zenodo.org/badge/latestdoi/663684836)
- [CLIP2Brain](https://github.com/ariaaay/clip2brain/tree/main)