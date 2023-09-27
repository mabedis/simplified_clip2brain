# Simplified CLIP2Brain Documentation

## About

## Installation

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
    * > `python3 main.py --action extract_cortical_voxel --mask_only --roi floc-bodies`

## Sources
