"""Generate `figures/20_sweep_gap_parallel_lc_combo.png` - `plot_gap_to_lc_parallel_coupler`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    PARALLEL_GAP_PLOT_CURRENT_GAP,
    PARALLEL_GAP_PLOT_CURRENT_LC,
    PARALLEL_RI,
    PARALLEL_SWEEP_EXTRA_MARGIN,
    PARALLEL_SWEEP_K_SEARCH,
    PARALLEL_SWEEP_LAMBDA_LAUNCH,
    PARALLEL_SWEEP_LENGTH_TOTAL,
    PARALLEL_SWEEP_MIN_FRACTION,
    PARALLEL_SWEEP_MIN_POWER,
    PARALLEL_SWEEP_SIGMA_RATIO,
    PARALLEL_TH,
    gap_values_parallel_sweep,
    workflow_notebook_config,
    x_parallel_sweep_mesh,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import WaveguideModalBPM1D, plot_gap_to_lc_parallel_coupler, sweep_gap_to_lc_parallel_coupler


def main() -> None:
    nb = workflow_notebook_config()
    analyzer = WaveguideModalBPM1D(nb)
    x_base = x_parallel_sweep_mesh()
    dx_base = float(x_base[1] - x_base[0])
    with silent_stdout():
        sweep = sweep_gap_to_lc_parallel_coupler(
            analyzer,
            x_base=x_base,
            dx_base=dx_base,
            refractive_indices=PARALLEL_RI,
            thicknesses=PARALLEL_TH,
            gap_values=gap_values_parallel_sweep(),
            lambda_launch=PARALLEL_SWEEP_LAMBDA_LAUNCH,
            length_total=PARALLEL_SWEEP_LENGTH_TOTAL,
            sigma_ratio=PARALLEL_SWEEP_SIGMA_RATIO,
            k_search=PARALLEL_SWEEP_K_SEARCH,
            extra_margin=PARALLEL_SWEEP_EXTRA_MARGIN,
            min_fraction=PARALLEL_SWEEP_MIN_FRACTION,
            min_power=PARALLEL_SWEEP_MIN_POWER,
        )
    fig, _ = plot_gap_to_lc_parallel_coupler(
        sweep,
        current_gap=PARALLEL_GAP_PLOT_CURRENT_GAP,
        current_lc=PARALLEL_GAP_PLOT_CURRENT_LC,
    )
    save_figure(fig, "20_sweep_gap_parallel_lc_combo.png")


if __name__ == "__main__":
    main()
