"""Generate `figures/19_sweep_gap_to_peak_transfer_coupler.png` - `plot_gap_to_peak_transfer_coupler`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    PLOT_SWEEP_GAP_TO_PEAK_COUPLER_TITLE_PREFIX,
    SWEEP_GAP_PEAK_COUPLER_CURRENT_GAP,
    SWEEP_PEAK_PLOT_AX,
    SWEEP_COUPLER_LAMBDA_LAUNCH,
    SWEEP_COUPLER_MIN_FRACTION,
    SWEEP_COUPLER_MIN_POWER,
    coupler_example_params,
    gap_values_coupler_sweep,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_gap_to_peak_transfer_coupler, sweep_gap_to_lc_coupler


def main() -> None:
    params = coupler_example_params()
    with silent_stdout():
        sweep = sweep_gap_to_lc_coupler(
            params=params,
            gap_values=gap_values_coupler_sweep(),
            lambda_launch=SWEEP_COUPLER_LAMBDA_LAUNCH,
            min_fraction=SWEEP_COUPLER_MIN_FRACTION,
            min_power=SWEEP_COUPLER_MIN_POWER,
        )
    fig, _ = plot_gap_to_peak_transfer_coupler(
        sweep,
        current_gap=SWEEP_GAP_PEAK_COUPLER_CURRENT_GAP,
        title_prefix=PLOT_SWEEP_GAP_TO_PEAK_COUPLER_TITLE_PREFIX,
        ax=SWEEP_PEAK_PLOT_AX,
    )
    save_figure(fig, "19_sweep_gap_to_peak_transfer_coupler.png")


if __name__ == "__main__":
    main()
