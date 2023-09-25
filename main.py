"""
This is the simplified version of CLIP2Brain.
"""

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--action',
        type=str,
        default='extract_image_list',
        choices=[
            'extract_image_list',
            'extract_cortical_voxel',
            # 'compute_ev',
            # 'extract_clip_features',
            # 'extract_features_across_models',
            # 'run_modeling',
            # 'analyze_clip_results',
            # 'visualize_in_pycortex',
            # 'analyze_clip_results',
            # 'analyze_clip_results_with_PCA',
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
        '-t', '--type',
        type=str,
        default='trial',
        choices=['trial', 'cocoId'],
        help='By default, `trial` will be chosen.',
    )
    parser.add_argument(
        '--mask_only',
        action="store_true",
        help="Only extract roi mask but not voxel response",
    )
    parser.add_argument(
        '--roi',
        type=str,
        default='floc-bodies',
        choices=[
            'prf-eccrois',
            'prf-visualrois',
            'floc-faces',
            'floc-words',
            'floc-places',
            'floc-bodies',
            'Kastner2015',
            'HCP_MMP1',
        ],
        help='By defaulf, `floc-bodies` will be chosen.',
    )
    args = parser.parse_args()

    if args.action == 'extract_image_list':
        print('extract_image_list ', args.subject, args.type)
    elif args.action == 'extract_cortical_voxel':
        print('extract_cortical_voxel', args.subject, args.mask_only, args.roi)
    # elif ...
