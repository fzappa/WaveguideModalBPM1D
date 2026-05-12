"""Generate `figures/05_straight_bpm_fundamental_heatmap.png` - `plot_straight_bpm_heatmap`.

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
    STRAIGHT_HEATMAP_CMAP,
    STRAIGHT_HEATMAP_TITLE,
    STRAIGHT_HEATMAP_VMAX_SCALE,
    STRAIGHT_HEATMAP_XLIM_FACTOR,
    workflow_notebook_config,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import WaveguideModalBPM1D, plot_straight_bpm_heatmap, run_straight_waveguide_bpm_fundamental


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
        result = run_straight_waveguide_bpm_fundamental(
            analyzer,
            modal_te,
            length_total=STRAIGHT_BPM_LENGTH,
            refractive_indices=NOTEBOOK_REFRACTIVE_INDICES,
            thicknesses=NOTEBOOK_THICKNESSES,
        )
    fig, _ = plot_straight_bpm_heatmap(
        result,
        title=STRAIGHT_HEATMAP_TITLE,
        cmap=STRAIGHT_HEATMAP_CMAP,
        vmax_scale=STRAIGHT_HEATMAP_VMAX_SCALE,
        xlim_factor=STRAIGHT_HEATMAP_XLIM_FACTOR,
    )
    save_figure(fig, "05_straight_bpm_fundamental_heatmap.png")


if __name__ == "__main__":
    main()
