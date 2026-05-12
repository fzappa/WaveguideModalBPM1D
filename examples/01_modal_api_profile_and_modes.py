"""Generate `figures/01_modal_api_profile_and_modes.png` - `plot_modal_analysis` (modal API).

Geometry and discretization: `example_parameters.py` (`MODAL_*` and `modal_structure_01`).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from example_parameters import modal_structure_01
from figure_helpers import ensure_repo_on_path, save_figure

ensure_repo_on_path()

from WaveguideModalBPM1D import analyze_modes, plot_modal_analysis


def main() -> None:
    res = analyze_modes(modal_structure_01())
    fig, _ = plot_modal_analysis(res)
    save_figure(fig, "01_modal_api_profile_and_modes.png")


if __name__ == "__main__":
    main()
