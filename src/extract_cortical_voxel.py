"""Extract Cortical Voxel."""

import os

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.stats import zscore

from src.utils import utils


class CorticalExtractionFactory:
    """Cortical Extraction functions."""

    def create_extractor(self, subject: int, all_subjects: bool, roi: str,
                         mask_only: bool, zscore_by_run: bool,
                         output_dir: str):
        """Create Cortical/Voxel extractor.

        Args:
            subject (int): Subject in range 1-8
            all_subjects (bool): To iterate over all the subjects.
            roi (str): Name of a valid region-of-interest.
            mask_only (bool): Only extract roi mask but not voxel response.
            zscore_by_run (bool): Z-Score brain data by runs.
        """
        utils.Directory(path=os.path.join(
            output_dir, 'output/cortical_voxels')).check_dir_existence()
        utils.Directory(path=os.path.join(
            output_dir, 'output/voxels_masks')).check_dir_existence()

        subjects = np.arange(1, 9) if all_subjects else [subject]

        if mask_only:
            return ExtractCorticalMask(
                subjects=subjects,
                roi=roi,
                output_dir=output_dir
            ).extract_corital_mask()
        return ExtractVoxels(
            subjects=subjects,
            roi=roi,
            zscore_by_run=zscore_by_run,
            output_dir=output_dir
        )


class ExtractCorticalMask:
    """Extract Cortical Mask from the subjects."""

    def __init__(self, subjects: list, roi: str, output_dir: str):
        self.subjects = subjects
        self.roi = roi
        self.output_dir = output_dir
        self._roi = utils.GetNSD(section='DATA', entry='PPdataPath')
        self._roi_path = self._roi.get_dataset_path()

    def extract_corital_mask(self):
        """Extract Cortical Mask.

        Returns:
            np.ndarray: Cortical Mask
        """
        for subj in self.subjects:
            directory = utils.Directory(path=os.path.join(
                self.output_dir, f"output/voxels_masks/subj{subj}"))
            directory.check_dir_existence()

            nsd_general_path = os.path.join(
                self._roi_path,
                f"subj{subj:02}/func1pt8mm/roi/nsdgeneral.nii.gz"
            )
            nsd_general = nib.load(nsd_general_path)
            nsd_cortical_mat = nsd_general.get_fdata()

            if self.roi in ("general", ""):
                anat_mat = nsd_cortical_mat
            else:
                roi_subj_path = os.path.join(
                    self._roi_path,
                    f"subj{subj:02}/func1pt8mm/roi/{self.roi}.nii.gz"
                )
                anat = nib.load(roi_subj_path)
                anat_mat = anat.get_fdata()

            roi_tag = "_" + self.roi if self.roi else ""
            if self.roi == "":  # cortical
                mask = anat_mat > -1
            else:  # roi
                mask = anat_mat > 0
                # save a 1D version as well
                cortical = nsd_cortical_mat > -1
                print("from NSD general, cortical voxel number is: "
                      f"{np.sum(cortical)}.")
                roi_1d_mask = anat_mat[cortical].astype(int)
                # assert np.sum(roi_1d_mask) == np.sum(mask)
                print("Number of non-zero ROI voxels: "
                      f"{str(np.sum(roi_1d_mask > 0))}")
                print("Number of cortical voxels is: "
                      f"{str(len(roi_1d_mask))}")
                assert len(roi_1d_mask) == np.sum(
                    cortical
                )  # check the roi 1D length is same as cortical numbers in NSD general
                np.save(
                    os.path.join(
                        self.output_dir,
                        f"output/voxels_masks/subj{subj}/roi_1d_mask_subj{subj:02}{roi_tag}.npy"
                    ),
                    roi_1d_mask,
                )

            np.save(
                os.path.join(
                    self.output_dir,
                    f"output/voxels_masks/subj{subj}/cortical_mask_subj{subj:02}{roi_tag}.npy"
                ),
                mask,
            )
            return mask


class ExtractVoxels:
    """Extract Voxels from the subjects."""

    def __init__(self, subjects: list, roi: str, zscore_by_run: bool,
                 output_dir: str, mask=None, mask_tag: str = ""):
        _beta = utils.GetNSD(section='DATA', entry='BetaPath')
        _beta_path = _beta.get_dataset_path()

        for subj in subjects:
            tag = roi
            zscore_tag = "zscored_by_run_" if zscore_by_run else ""
            output_path = os.path.join(
                output_dir,
                f"output/cortical_voxels/cortical_voxel_across_sessions_{zscore_tag}"
                f"subj{subj:02}{mask_tag}.npy"
            )

            beta_subj_dir = os.path.join(
                _beta_path,
                f"subj{subj:02}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
            )
            if mask is None:
                try:
                    mask = np.load(os.path.join(
                        output_dir,
                        f"output/voxels_masks/subj{subj}/cortical_mask_subj{subj:02}{tag}.npy"
                    ))
                except FileNotFoundError:
                    mask = ExtractCorticalMask(
                        subjects=subjects, roi=roi, output_dir=output_dir
                    ).extract_corital_mask()

            cortical_beta_mat = None
            # NOTE: Just for test, otherwise it should be changed to (1, 9)
            for ses in tqdm(range(1, 3)):
                try:
                    beta_file = nib.load(
                        os.path.join(
                            beta_subj_dir,
                            f"betas_session{ses:02}.nii.gz"
                        )
                    )
                except FileNotFoundError as err:
                    print(f'{err}')
                    break
                beta = beta_file.get_fdata()
                cortical_beta = (beta[mask]).T  # verify the mask with array

                if cortical_beta_mat is None:
                    cortical_beta_mat = cortical_beta / 300
                else:
                    cortical_beta_mat = np.vstack(
                        (cortical_beta_mat, cortical_beta / 300))

            print(f"NaN Values: {str(np.any(np.isnan(cortical_beta_mat)))}")
            print(f"Is finite: {str(np.all(np.isfinite(cortical_beta_mat)))}")

            if zscore_by_run:
                print("Zscoring...")
                cortical_beta_mat = self.__zscore_by_run(cortical_beta_mat)
                finite_flag = np.all(np.isfinite(cortical_beta_mat))
                print(f"Is finite: {str(finite_flag)}")

                if finite_flag is False:
                    nonzero_mask = (
                        np.sum(np.isfinite(cortical_beta_mat), axis=0)
                        == cortical_beta_mat.shape[0]
                    )
                    np.save(
                        os.path.join(
                            output_dir,
                            f"output/voxels_masks/subj{subj}/nonzero_voxels_subj{subj:02}.npy"
                        ),
                        nonzero_mask,
                    )

            np.save(output_path, cortical_beta_mat)

    def __zscore_by_run(self, mat, run_n: int = 480):
        run_n = np.ceil(
            mat.shape[0] / 62.5
        )  # should be 480 for subject with full experiment\

        zscored_mat = np.zeros(mat.shape)
        index_so_far = 0
        for i in tqdm(range(int(run_n))):
            if i % 2 == 0:
                zscored_mat[index_so_far: index_so_far + 62, :] = zscore(
                    mat[index_so_far: index_so_far + 62, :]
                )
                index_so_far += 62
            else:
                zscored_mat[index_so_far: index_so_far + 63, :] = zscore(
                    mat[index_so_far: index_so_far + 63, :]
                )
                index_so_far += 63

        return zscored_mat
