"""The utility functions for any mid-way computations."""

import numpy as np


class Computations:
    """Class for all computations."""

    def compute_ev(self, data: np.array, bias_corr: bool = True) -> float:
        """
        Computes the amount of variance in a voxel's response that can be
        explained by the mean response of that voxel over multiple repetitions
        of the same stimulus.

        Args:
            data (np.array): Data is assumed to be a 2D matrix: time x repeats.
            bias_corr (bool, optional): If [bias_corr], the explainable variance
                is corrected for bias, and will have mean zero for random datasets.
                Defaults to True.

        Returns:
            float: bias_corr or ev_value.
        """
        ev = 1 - np.nanvar(data.T - np.nanmean(data, axis=1)) / np.nanvar(data)
        return ev - ((1 - ev) / (data.shape[1] - 1.0)) if bias_corr else ev
