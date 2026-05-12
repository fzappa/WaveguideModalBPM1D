"""Generate `figures/13_coupler_power_and_profiles.png` - `plot_coupler_transfer_and_profiles`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import SIMULATE_COUPLER_LAMBDA_LAUNCH, coupler_example_params
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_coupler_transfer_and_profiles, simulate_coupler


def main() -> None:
    params = coupler_example_params()
    with silent_stdout():
        sim = simulate_coupler(params=params, lambda_launch=SIMULATE_COUPLER_LAMBDA_LAUNCH)
    fig, _ = plot_coupler_transfer_and_profiles(sim, params)
    save_figure(fig, "13_coupler_power_and_profiles.png")


if __name__ == "__main__":
    main()
