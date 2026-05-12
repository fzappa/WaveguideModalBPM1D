"""Generate `figures/09_parallel_coupler_power.png` - `plot_parallel_power_transfer`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    PARALLEL_GAP,
    PARALLEL_K_SEARCH,
    PARALLEL_LAMBDA_LAUNCH,
    PARALLEL_LENGTH_TOTAL,
    PARALLEL_RI,
    PARALLEL_SIGMA_RATIO,
    PARALLEL_TH,
    x_parallel_coupler_mesh,
    workflow_notebook_config,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import WaveguideModalBPM1D, plot_parallel_power_transfer, run_parallel_coupler_bpm


def main() -> None:
    nb = workflow_notebook_config()
    analyzer = WaveguideModalBPM1D(nb)
    x = x_parallel_coupler_mesh()
    dx = float(x[1] - x[0])
    with silent_stdout():
        t3 = run_parallel_coupler_bpm(
            analyzer,
            x=x,
            dx=dx,
            refractive_indices=PARALLEL_RI,
            thicknesses=PARALLEL_TH,
            gap=PARALLEL_GAP,
            lambda_launch=PARALLEL_LAMBDA_LAUNCH,
            length_total=PARALLEL_LENGTH_TOTAL,
            sigma_ratio=PARALLEL_SIGMA_RATIO,
            k_search=PARALLEL_K_SEARCH,
        )
    fig, _ = plot_parallel_power_transfer(t3)
    save_figure(fig, "09_parallel_coupler_power.png")


if __name__ == "__main__":
    main()
