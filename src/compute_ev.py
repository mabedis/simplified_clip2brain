"""
Compute explainable variance for the data and output data averaged by repeats.
"""

import os
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import utils
from src.utils.computations import Computations
from src.utils.model_config import roi_name_dict


class EVFactory:
    """Compute EV."""

    def create_ev(
        self,
        subject: int,
        zscored_input: bool,
        compute_ev: bool,
        roi_only: bool,
        bias_corr: bool,
        compute_sample_ev: bool,
        roi_for_sample_ev: str,
        output_dir: str,
    ) -> None:
        """Preprocessing the given arguments before computation."""
        utils.Directory(path=os.path.join(
            output_dir, 'output/figures')).check_dir_existence()

        tag: str = ""
        roi = '_roi_only' if roi_only else ''
        tag += roi
        tag += '_biascorr' if bias_corr else ''
        tag += '_zscored' if zscored_input else ''

        if compute_ev:
            print("Computing EVs...")
            all_evs = ComputeEV(
                subject=subject,
                output_dir=output_dir,
                roi=roi,
                bias_corr=bias_corr,
                zscored_input=zscored_input,
            ).compute_ev()
            np.save(
                os.path.join(
                    output_dir,
                    f"output/evs_subj{subject:02}{tag}.npy"
                ),
                all_evs
            )

            plt.figure()
            plt.hist(all_evs)
            plt.title(
                f"Explainable Variance across Voxels (subj{subject:02}{tag})")
            plt.savefig(f"output/figures/evs_subj{subject:02}{tag}.png")

        elif compute_sample_ev:
            sample_ev_by_roi: dict = {}
            roi = roi_for_sample_ev
            if roi is not None:
                roi_mask = np.load(
                    os.path.join(
                        output_dir,
                        f"output/voxels_masks/subj{subject:01}",
                    ),
                    f"output/roi_1d_mask_subj{subject:02}_{roi}.npy"
                )
                roi_dict = roi_name_dict[roi]
                for k, v in roi_dict.items():
                    if k > 0:
                        mask = roi_mask == k
                        sample_ev = ComputeEV(
                            subject=subject,
                            output_dir=output_dir,
                            mask=mask,
                            roi=roi,
                            bias_corr=bias_corr,
                            zscored_input=zscored_input,
                        ).compute_sample_wise_ev()
                        sample_ev_by_roi[v] = sample_ev
                json.dump(
                    sample_ev_by_roi,
                    open(
                        file=(f"{output_dir}/sample_snr/"
                              f"sample_snr_subj{subject:02}_{roi}.json"),
                        mode="w",
                        encoding='UTF8'
                    ),
                )

            else:
                sample_evs = ComputeEV(
                    subject=subject,
                    output_dir=output_dir,
                    roi=roi,
                    bias_corr=bias_corr,
                    zscored_input=zscored_input,
                ).compute_sample_wise_ev()
                np.save(
                    os.path.join(
                        output_dir,
                        "output/sample_snr/sample_snr_subj"
                        f"{subject:02}_{roi}.npy",
                    ),
                    sample_evs,
                )


class ComputeEV:
    """Compute EV Class."""

    def __init__(
        self,
        subject: int,
        output_dir: str,
        mask: int = 0,
        roi: str = "",
        bias_corr: bool = False,
        zscored_input: bool = False,
    ):
        self.subject = subject
        self.roi = roi
        self.mask = mask
        self.bias_corr = bias_corr
        self.zscored_input = zscored_input
        self.output_dir = output_dir

    def compute_ev(self) -> np.array:
        """Compute EV.

        Returns:
            np.array: EV list.
        """
        l = np.load(
            os.path.join(
                self.output_dir,
                f"output/trials_subj{self.subject:02}.npy"
            )
        )  # size should be 10000 by 3 for subj 1,2,5,7; ordered by image id. entries are trial numbers

        repeat_n = l.shape[0]
        # print("The number of images with 3 repetitions are: " + str(repeat_n))

        try:
            assert l.shape == (repeat_n, 3)
        except AssertionError:
            print("Irregular trial shape:")
            print(l.shape)

        if self.zscored_input:
            data = np.load(
                os.path.join(
                    self.output_dir,
                    f"output/cortical_voxels/cortical_voxel_across_sessions"
                    f"_zscored_by_run_subj{self.subject:02}{self.roi}.npy"
                )
            )
        else:
            data = np.load(
                os.path.join(
                    self.output_dir,
                    "output/cortical_voxels/cortical_voxel_across"
                    f"_sessions_subj{self.subject:02}{self.roi}.npy"
                )
            )

        # data size is # of total trials X # of voxels
        ev_list = []
        avg_mat = np.zeros(
            (repeat_n, data.shape[1])
        )  # size = number of repeated images by number of voxels

        print(f"Brain data shape is: {data.shape}")
        # import pdb; pdb.set_trace()
        # fill in 0s for nonexisting trials
        if data.shape[0] < 30000:
            tmp = np.zeros((30000, data.shape[1]))
            tmp[:] = np.nan
            tmp[: data.shape[0], :] = data.copy()
            data = tmp

        for v in tqdm(range(data.shape[1])):  # loop over voxels
            repeat = []
            for r in range(3):
                try:
                    # all repeated trials for each voxels
                    repeat.append(data[l[:, r], v])
                except IndexError:
                    print("Index Error")
                    print(r, v)

            # repeat size: 3 x # repeated images
            repeat = np.array(repeat).T
            try:
                assert repeat.shape == (repeat_n, 3)
                avg_mat[:, v] = np.nanmean(repeat, axis=1)
                # print("NaNs:")
                # print(np.sum(np.isnan(avg_mat[:, v])))
            except AssertionError:
                print(repeat.shape)

            ev_list.append(Computations.compute_ev(
                self, data=repeat, bias_corr=self.bias_corr)
            )
        np.save(
            os.path.join(
                self.output_dir,
                "output/cortical_voxels/averaged_cortical_responses_zscored"
                f"_by_run_subj{self.subject:02}{self.roi}.npy"
            ),
            avg_mat,
        )
        return np.array(ev_list)

    def compute_sample_wise_ev(self) -> list:
        """Compute sample-wise EV.

        Returns:
            list: ev_list
        """
        l = np.load(
            os.path.join(
                self.output_dir,
                f"output/trials_subj{self.subject:02}.npy"
            )
        )  # size should be 10000 by 3 for subj 1,2,5,7; ordered by image id

        repeat_n = l.shape[0]
        print(f"The number of images with 3 repetitions are: {str(repeat_n)}")

        try:
            assert l.shape == (repeat_n, 3)
        except AssertionError:
            print(l.shape)

        if self.zscored_input:
            data = np.load(
                os.path.join(
                    self.output_dir,
                    "output/cortical_voxels/cortical_voxel_across_sessions"
                    f"_zscored_by_run_subj{self.subject:02}.npy"
                )
            )
        else:
            data = np.load(
                os.path.join(
                    self.output_dir,
                    "output/cortical_voxels/"
                    f"cortical_voxel_across_sessions_subj{self.subject:02}.npy"
                )
            )

        data = data[:, self.mask]   # index by roi
        print(f"Brain data shape is: {data.shape}")

        ev_list = []
        for i in tqdm(range(l.shape[0])):  # loop over images
            repeat = data[l[i, :], :].T  # all repeated trials for each voxels
            assert repeat.shape == (data.shape[1], 3)
            ev_list.append(Computations.compute_ev(
                self, data=repeat, bias_corr=self.bias_corr))

        return ev_list


# def extract_subject_trials_index_shared1000(stim, subj):
#     """Extract each subject's index for 1000 images."""
#     index = []
#     for i in range(3):
#         col = f"subject{subj:01}_rep{i:01}"
#         assert len(stim[col][stim["shared1000"]]) == 1000
#         index.append(list(stim[col][stim["shared1000"]]))
#     assert len(index) == 3
#     return index
