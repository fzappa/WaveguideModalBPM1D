"""Generate `figures/04_workflow_dispersion_with_angle_axis.png` - `plot_dispersion_with_angle_axis`.

Parameters: `example_parameters.py` (`NOTEBOOK_*`, `DISPERSION_*`, `SOLVE_*`).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    DISPERSION_MAX_MODES_CAP,
    DISPERSION_N_X,
    DISPERSION_POLARIZATION,
    dispersion_d_norm_vec,
    workflow_notebook_config,
)
from figure_helpers import ensure_repo_on_path, save_figure

ensure_repo_on_path()

from WaveguideModalBPM1D import WaveguideModalBPM1D, plot_dispersion_with_angle_axis


def main() -> None:
    nb = workflow_notebook_config()
    analyzer = WaveguideModalBPM1D(nb)
    disp = analyzer.compute_dispersion(
        dispersion_d_norm_vec(),
        polarization=DISPERSION_POLARIZATION,
        n_x=DISPERSION_N_X,
        max_modes_cap=DISPERSION_MAX_MODES_CAP,
    )
    fig, _ = plot_dispersion_with_angle_axis(
        d_norm_vec=disp["d_norm_vec"],
        neff_matrix=disp["neff_matrix"],
        max_modes=disp["max_modes"],
        n_clad_edge=float(disp["n_clad_edge"]),
        n_core=float(disp["n_core"]),
        lambda_0=analyzer.lambda_0,
    )
    save_figure(fig, "04_workflow_dispersion_with_angle_axis.png")


if __name__ == "__main__":
    main()
