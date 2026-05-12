"""Generate `figures/06_straight_bpm_input_output_profiles.png` - `plot_straight_profile_comparison`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

from example_parameters import (
    NOTEBOOK_N_POINTS,
    NOTEBOOK_REFRACTIVE_INDICES,
    NOTEBOOK_THICKNESSES,
    SOLVE_K_SEARCH,
    SOLVE_N_CLADDING,
    SOLVE_POLARIZATION,
    STRAIGHT_BPM_LENGTH,
    STRAIGHT_GAUSSIAN_TITLE,
    STRAIGHT_GAUSSIAN_SIGMA,
    STRAIGHT_GAUSSIAN_XC_INPUT,
    STRAIGHT_PROFILE_XLIM_FACTOR,
    workflow_notebook_config,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import WaveguideModalBPM1D, plot_straight_profile_comparison, run_straight_waveguide_bpm_gaussian


def main() -> None:
    nb = workflow_notebook_config()
    analyzer = WaveguideModalBPM1D(nb)
    modal_te = analyzer.solve_modes(
        polarization=SOLVE_POLARIZATION,
        thicknesses=NOTEBOOK_THICKNESSES,
        n_points=NOTEBOOK_N_POINTS,
        n_cladding=SOLVE_N_CLADDING,
        lambda_launch=None,
        indices=np.asarray(NOTEBOOK_REFRACTIVE_INDICES, dtype=float),
        k_search=SOLVE_K_SEARCH,
    )
    with silent_stdout():
        result = run_straight_waveguide_bpm_gaussian(
            analyzer,
            modal_te,
            length_total=STRAIGHT_BPM_LENGTH,
            refractive_indices=NOTEBOOK_REFRACTIVE_INDICES,
            thicknesses=NOTEBOOK_THICKNESSES,
            xc_input=STRAIGHT_GAUSSIAN_XC_INPUT,
            sigma=STRAIGHT_GAUSSIAN_SIGMA,
        )
    fig, _ = plot_straight_profile_comparison(
        result,
        title=STRAIGHT_GAUSSIAN_TITLE,
        xlim_factor=STRAIGHT_PROFILE_XLIM_FACTOR,
    )
    save_figure(fig, "06_straight_bpm_input_output_profiles.png")


if __name__ == "__main__":
    main()
