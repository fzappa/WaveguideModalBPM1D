"""Generate `figures/03_workflow_multilayer_modes.png` - `plot_multilayer_modes`.

`NotebookConfig`, `solve_modes`, and plot parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np

from example_parameters import (
    MULTILAYER_FIELD_SYMBOL,
    MULTILAYER_PLOT_POLARIZATION,
    NOTEBOOK_LAMBDA_0,
    NOTEBOOK_N_POINTS,
    NOTEBOOK_REFRACTIVE_INDICES,
    NOTEBOOK_THICKNESSES,
    SOLVE_K_SEARCH,
    SOLVE_N_CLADDING,
    SOLVE_POLARIZATION,
    workflow_notebook_config,
)
from figure_helpers import ensure_repo_on_path, save_figure

ensure_repo_on_path()

from WaveguideModalBPM1D import WaveguideModalBPM1D, plot_multilayer_modes


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
    geo = analyzer.build_multilayer_geometry(
        thicknesses=NOTEBOOK_THICKNESSES,
        n_points=NOTEBOOK_N_POINTS,
    )
    fig, _ = plot_multilayer_modes(
        x=geo["x"],
        n_profile=geo["n_profile"],
        boundaries=geo["boundaries"],
        refractive_indices=NOTEBOOK_REFRACTIVE_INDICES,
        thicknesses=NOTEBOOK_THICKNESSES,
        lambda_0=NOTEBOOK_LAMBDA_0,
        modes=modal_te["modes"],
        n_effs=modal_te["n_effs"],
        polarization=MULTILAYER_PLOT_POLARIZATION,
        field_symbol=MULTILAYER_FIELD_SYMBOL,
    )
    save_figure(fig, "03_workflow_multilayer_modes.png")


if __name__ == "__main__":
    main()
