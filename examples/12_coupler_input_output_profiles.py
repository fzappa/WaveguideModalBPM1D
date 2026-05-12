"""Generate `figures/12_coupler_input_output_profiles.png` - `plot_coupler_output_profiles`.

Parameters: `example_parameters.py`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import (
    COUPLER_PROFILES_AX,
    COUPLER_PROFILES_TITLE,
    COUPLER_PROFILES_UNIT_LABEL,
    COUPLER_PROFILES_UNIT_SCALE,
    SIMULATE_COUPLER_LAMBDA_LAUNCH,
    coupler_example_params,
)
from figure_helpers import ensure_repo_on_path, save_figure, silent_stdout

ensure_repo_on_path()

from WaveguideModalBPM1D import plot_coupler_output_profiles, simulate_coupler


def main() -> None:
    params = coupler_example_params()
    with silent_stdout():
        sim = simulate_coupler(params=params, lambda_launch=SIMULATE_COUPLER_LAMBDA_LAUNCH)
    fig, _ = plot_coupler_output_profiles(
        sim,
        params,
        ax=COUPLER_PROFILES_AX,
        unit_scale=COUPLER_PROFILES_UNIT_SCALE,
        unit_label=COUPLER_PROFILES_UNIT_LABEL,
        title=COUPLER_PROFILES_TITLE,
    )
    save_figure(fig, "12_coupler_input_output_profiles.png")


if __name__ == "__main__":
    main()
