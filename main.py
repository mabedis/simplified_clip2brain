"""
This is the simplified version of CLIP2Brain.
"""

import argparse

from src.compute_ev import EVFactory
from src.extract_clip_features import ExtractFeatures
from src.extract_image_list import ExtractImageFactory
from src.extract_cortical_voxel import CorticalExtractionFactory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action',
        type=str,
        default='extract_image_list',
        choices=[
            'extract_image_list',
            'extract_cortical_voxel',
            'compute_ev',
            'extract_clip_features',
            'extract_features_across_models',
            'run_modeling',
            'analyze_clip_results',
            'visualize_in_pycortex',
            'analyze_clip_results_with_PCA',
        ],
        help=('Choose an action against the dataset. By default, '
              '`extract_image_list` will be chosen.'),
    )
    parser.add_argument(
        '-s', '--subject',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help='By default, `1` will be chosen',
    )
    parser.add_argument(
        "--all_subj",
        default=False,
        action="store_true",
        help="Extract cortical voxel for all subjects.",
    )
    parser.add_argument(
        '-t', '--type',
        type=str,
        default='trial',
        choices=['trial', 'cocoId'],
        help='By default, `trial` will be chosen.',
    )
    parser.add_argument(
        '--zscore_by_run',
        default=False,
        action='store_true',
        help='Z-Score brain data by runs.',
    )
    parser.add_argument(
        '--mask_only',
        action="store_true",
        help="Only extract roi mask but not voxel response.",
    )
    parser.add_argument(
        '--roi',
        type=str,
        default='floc-bodies',
        choices=[
            'prf-eccrois',
            'prf-visualrois',
            'corticalsulc',
            'floc-bodies',
            'floc-faces',
            'floc-places',
            'floc-words',
            'HCP_MMP1',
            'Kastner2015',
            'lh.corticalsulc',
            'lh.floc-bodies',
            'lh.floc-faces',
            'lh.floc-places',
            'lh.floc-words',
            'lh.HCP_MMP1',
            'lh.Kastner2015',
            'lh.MTL',
            'lh.nsdgeneral',
            'lh.prf-eccrois',
            'lh.prf-visualrois',
            'lh.streams',
            'lh.thalamus',
            'MTL',
            'nsdgeneral',
            'prf-eccrois',
            'prf-visualrois',
            'rh.corticalsulc',
            'rh.floc-bodies',
            'rh.floc-faces',
            'rh.floc-places',
            'rh.floc-words',
            'rh.HCP_MMP1',
            'rh.Kastner2015',
            'rh.MTL',
            'rh.nsdgeneral',
            'rh.prf-eccrois',
            'rh.prf-visualrois',
            'rh.streams',
            'rh.thalamus',
            'streams',
            'thalamus',
        ],
        help=('By defaulf, `floc-bodies` will be chosen.\n'
              'Input arguments are the ROIs\' file names in '
              'natural-scenes-dataset/nsddata/ppdata/subj{01}/func1pt8mm/roi'),
    )
    parser.add_argument(
        '--zscored_input',
        action='store_true',
        help='',
    )
    parser.add_argument(
        '--compute_ev',
        action='store_true',
        help='',
    )
    parser.add_argument(
        '--roi_only',
        action='store_true',
        help='',
    )
    parser.add_argument(
        '--bias_corr',
        action='store_true',
        help='',
    )
    parser.add_argument(
        '--compute_sample_ev',
        action='store_true',
        help='',
    )
    parser.add_argument(
        '--roi_for_sample_ev',
        type=str,
        default=None,
        help='',
    )
    parser.add_argument(
        '--feature_dir',
        type=str,
        default='.',
        help='',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='By default, project\'s root directory will be the destination.'
    )
    args = parser.parse_args()

    if args.action == 'extract_image_list':
        ExtractImageFactory.create_extractor(
            self=None,
            subject=args.subject,
            image_type=args.type,
            output_dir=args.output_dir,
        )
    elif args.action == 'extract_cortical_voxel':
        CorticalExtractionFactory.create_extractor(
            self=None,
            subject=args.subject,
            all_subjects=args.all_subj,
            roi=args.roi,
            mask_only=args.mask_only,
            zscore_by_run=args.zscore_by_run,
            output_dir=args.output_dir,
        )
    elif args.action == 'compute_ev':
        EVFactory.create_ev(
            self=None,
            subject=args.subject,
            zscored_input=args.zscored_input,
            compute_ev=args.compute_ev,
            roi_only=args.roi_only,
            bias_corr=args.bias_corr,
            compute_sample_ev=args.compute_sample_ev,
            roi_for_sample_ev=args.roi_for_sample_ev,
            output_dir=args.output_dir,
        )
    elif args.action == 'extract_clip_features':
        ExtractFeatures(
            subject=args.subject,
            output_dir=args.output_dir,
            feature_dir=args.feature_dir,
        )
    # elif args.action == 'extract_features_across_models':
    #     print('extract_features_across_models',
    #           args.subject, args.mask_only, args.roi)
    # elif args.action == 'run_modeling':
    #     print('run_modeling', args.subject, args.mask_only, args.roi)
    # elif args.action == 'analyze_clip_results':
    #     print('analyze_clip_results', args.subject, args.mask_only, args.roi)
    # elif args.action == 'visualize_in_pycortex':
    #     print('visualize_in_pycortex', args.subject, args.mask_only, args.roi)
    # elif args.action == 'analyze_clip_results_with_PCA':
    #     print('analyze_clip_results_with_PCA',
    #           args.subject, args.mask_only, args.roi)
    else:
        raise ValueError(f'Not a valid action: {args.action}')
