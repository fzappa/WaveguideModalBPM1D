"""
Explicit numerical and geometric parameters for the figure gallery.

The `NN_*.py` scripts import from here so they do not rely on omitted arguments
in the WaveguideModalBPM1D package classes/functions (see library defaults in the README).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from WaveguideModalBPM1D import CouplerTestParams, ModalStructureParams, NotebookConfig, nm, um

# --- Modal API: 5-layer stack (examples 01 and 02) ---
MODAL_REFRACTIVE_INDICES_5 = (3.55, 3.55, 3.60, 3.55, 3.55)
MODAL_THICKNESSES_5 = (
    5.0 * um,
    0.5 * um,
    1.0 * um,
    0.5 * um,
    5.0 * um,
)
MODAL_LAMBDA_0 = 1.0 * um
MODAL_N_POINTS_01 = 1000
MODAL_N_POINTS_02 = 1000

# Modal dispersion (02): normalized d/lambda sweep
MODAL_DISPERSION_D_MIN = 0.15
MODAL_DISPERSION_D_MAX = 3.0
MODAL_DISPERSION_D_COUNT = 36
MODAL_DISPERSION_MAX_MODES_CAP = 8

# --- Notebook + 3-layer waveguide (03-06, 07-09, analyzer base in 20-22) ---
NOTEBOOK_LAMBDA_0 = 1.0 * um
NOTEBOOK_N_POINTS = 1000
NOTEBOOK_REFRACTIVE_INDICES = (3.45, 3.48, 3.45)
NOTEBOOK_THICKNESSES = (4.0 * um, 1.0 * um, 4.0 * um)
NOTEBOOK_JOBLIB_N_JOBS = 1
NOTEBOOK_JOBLIB_BACKEND = "loky"
NOTEBOOK_JOBLIB_VERBOSE = 0

# solve_modes / analyzer-based dispersion
SOLVE_POLARIZATION = "TE"
SOLVE_K_SEARCH = 6
SOLVE_N_CLADDING = 3.45  # min(left cladding, right cladding) for this symmetric profile

# compute_dispersion (04)
DISPERSION_D_NORM_MIN = 0.1
DISPERSION_D_NORM_MAX = 2.5
DISPERSION_D_NORM_COUNT = 32
DISPERSION_N_X = 500
DISPERSION_MAX_MODES_CAP = 8
DISPERSION_POLARIZATION = "TE"

# Straight-waveguide BPM (05, 06)
STRAIGHT_BPM_LENGTH = 80.0 * um
STRAIGHT_HEATMAP_TITLE = "Straight-waveguide BPM: fundamental-mode excitation"
STRAIGHT_HEATMAP_CMAP = "inferno"
STRAIGHT_HEATMAP_VMAX_SCALE = 1.0
STRAIGHT_HEATMAP_XLIM_FACTOR = 5.0
STRAIGHT_GAUSSIAN_TITLE = "Straight-waveguide BPM: Gaussian vs theoretical output mode"
STRAIGHT_PROFILE_XLIM_FACTOR = 2.0
STRAIGHT_GAUSSIAN_XC_INPUT = 0.0
# Core thickness (highest-index layer): 0.8 um - sigma = half of it (same rule as the Gaussian BPM default)
STRAIGHT_GAUSSIAN_SIGMA = 0.5 * 0.8 * um

# plot_multilayer_modes (03)
MULTILAYER_PLOT_POLARIZATION = "TE"
MULTILAYER_FIELD_SYMBOL = "E"

# --- Parallel coupler / T3 (07-09) ---
PARALLEL_RI = (1.45, 1.50, 1.45)
PARALLEL_TH = (5.0 * um, 0.5 * um, 5.0 * um)
PARALLEL_X_HALF_WIDTH_M = 15e-6
PARALLEL_N_X = 601
PARALLEL_GAP = 1.0 * um
PARALLEL_LAMBDA_LAUNCH = 1.550 * um
PARALLEL_LENGTH_TOTAL = 2000.0 * um
PARALLEL_SIGMA_RATIO = 0.45
PARALLEL_K_SEARCH = 6
PARALLEL_PLOT_XLIM_FACTOR = 4.0

# --- Generic coupler (10-14, 15-19): all CouplerTestParams fields ---
COUPLER_N_CORE = 1.50
COUPLER_N_CLAD = 1.42
COUPLER_WINDOW = 40.0 * um
COUPLER_LAMBDA_LAUNCH = 1.0 * um
COUPLER_LENGTH_TOTAL = 900.0 * um
COUPLER_LENGTH_COUPLING = 450.0 * um
COUPLER_PORT_OFFSET = 10.0 * um
COUPLER_CORE_WIDTH = 1.0 * um
COUPLER_CENTER_OFFSET = 0.75 * um
COUPLER_SIGMA = 0.45 * um
COUPLER_N_POINTS = 241
COUPLER_MODE_SEARCH_COUNT = 8

# Coupler gap sweep (15-19)
SWEEP_COUPLER_GAP_MIN = 0.4 * um
SWEEP_COUPLER_GAP_MAX = 1.8 * um
SWEEP_COUPLER_GAP_COUNT = 4
SWEEP_COUPLER_LAMBDA_LAUNCH: float | None = None  # uses params.lambda_launch
SWEEP_COUPLER_MIN_FRACTION = 0.05
SWEEP_COUPLER_MIN_POWER = 0.05

# Sweep labels / figures
PLOT_SWEEP_PARAMETER_TO_LC_TITLE_PREFIX = "Sweep (coupler)"
PLOT_SWEEP_PARAMETER_LC_TITLE_PREFIX = "Lc vs gap (generic)"
PLOT_SWEEP_PARAMETER_PEAK_TITLE_PREFIX = "Peak vs gap (generic)"
PLOT_SWEEP_GAP_TO_LC_COUPLER_TITLE_PREFIX = "Coupler"
PLOT_SWEEP_GAP_TO_PEAK_COUPLER_TITLE_PREFIX = "Coupler"
PLOT_PARAMETER_KEY = "gap"
PLOT_PARAMETER_LABEL = "gap"
PLOT_PARAMETER_UNIT_SCALE = um
PLOT_PARAMETER_UNIT_LABEL = "um"
PLOT_PARAMETER_TO_LC_FIGSIZE = (14.0, 5.0)
PLOT_PARAMETER_TO_LC_CURRENT_PARAMETER = None
PLOT_PARAMETER_TO_LC_CURRENT_LC = None

SWEEP_LC_CURRENT_PARAMETER = None
SWEEP_LC_CURRENT_LC = None

SWEEP_GAP_COUPLER_CURRENT_GAP = None
SWEEP_GAP_COUPLER_CURRENT_LC = None

# Spectral response (14)
WAVELENGTH_SWEEP_MIN = 0.92 * um
WAVELENGTH_SWEEP_MAX = 1.08 * um
WAVELENGTH_SWEEP_COUNT = 6
WAVELENGTH_SWEEP_N_JOBS = 1
WAVELENGTH_SWEEP_BACKEND = "loky"
WAVELENGTH_SWEEP_VERBOSE = 0
WAVELENGTH_PLOT_TITLE = "Spectral response of the output ports"  # matches the default in `plot_coupler_wavelength_response`
WAVELENGTH_PLOT_UNIT_SCALE = nm
WAVELENGTH_PLOT_UNIT_LABEL = "nm"

# Parallel sweep (20-22)
PARALLEL_SWEEP_X_HALF_M = 18e-6
PARALLEL_SWEEP_N_X = 501
PARALLEL_SWEEP_GAP_MIN = 0.5 * um
PARALLEL_SWEEP_GAP_MAX = 2.0 * um
PARALLEL_SWEEP_GAP_COUNT = 4
PARALLEL_SWEEP_LAMBDA_LAUNCH = 1.55 * um
PARALLEL_SWEEP_LENGTH_TOTAL = 1000.0 * um
PARALLEL_SWEEP_SIGMA_RATIO = 0.45
PARALLEL_SWEEP_K_SEARCH = 6
PARALLEL_SWEEP_EXTRA_MARGIN = 5.0 * um
PARALLEL_SWEEP_MIN_FRACTION = 0.05
PARALLEL_SWEEP_MIN_POWER = 0.05


def modal_structure_01() -> ModalStructureParams:
    return ModalStructureParams(
        lambda_0=MODAL_LAMBDA_0,
        n_points=MODAL_N_POINTS_01,
        refractive_indices=MODAL_REFRACTIVE_INDICES_5,
        thicknesses=MODAL_THICKNESSES_5,
    )


def modal_structure_02() -> ModalStructureParams:
    return ModalStructureParams(
        lambda_0=MODAL_LAMBDA_0,
        n_points=MODAL_N_POINTS_02,
        refractive_indices=MODAL_REFRACTIVE_INDICES_5,
        thicknesses=MODAL_THICKNESSES_5,
    )


def modal_dispersion_d_norm_vec() -> np.ndarray:
    return np.linspace(MODAL_DISPERSION_D_MIN, MODAL_DISPERSION_D_MAX, MODAL_DISPERSION_D_COUNT)


def workflow_notebook_config() -> NotebookConfig:
    return NotebookConfig(
        lambda_0=NOTEBOOK_LAMBDA_0,
        n_points=NOTEBOOK_N_POINTS,
        refractive_indices=NOTEBOOK_REFRACTIVE_INDICES,
        thicknesses=NOTEBOOK_THICKNESSES,
        joblib_n_jobs=NOTEBOOK_JOBLIB_N_JOBS,
        joblib_backend=NOTEBOOK_JOBLIB_BACKEND,
        joblib_verbose=NOTEBOOK_JOBLIB_VERBOSE,
    )


def dispersion_d_norm_vec() -> np.ndarray:
    return np.linspace(DISPERSION_D_NORM_MIN, DISPERSION_D_NORM_MAX, DISPERSION_D_NORM_COUNT)


def coupler_example_params() -> CouplerTestParams:
    return CouplerTestParams(
        n_core=COUPLER_N_CORE,
        n_clad=COUPLER_N_CLAD,
        window=COUPLER_WINDOW,
        lambda_launch=COUPLER_LAMBDA_LAUNCH,
        length_total=COUPLER_LENGTH_TOTAL,
        length_coupling=COUPLER_LENGTH_COUPLING,
        port_offset=COUPLER_PORT_OFFSET,
        core_width=COUPLER_CORE_WIDTH,
        center_offset=COUPLER_CENTER_OFFSET,
        sigma=COUPLER_SIGMA,
        n_points=COUPLER_N_POINTS,
        mode_search_count=COUPLER_MODE_SEARCH_COUNT,
    )


def gap_values_coupler_sweep() -> np.ndarray:
    return np.linspace(SWEEP_COUPLER_GAP_MIN, SWEEP_COUPLER_GAP_MAX, SWEEP_COUPLER_GAP_COUNT)


def wavelength_values_sweep() -> np.ndarray:
    return np.linspace(WAVELENGTH_SWEEP_MIN, WAVELENGTH_SWEEP_MAX, WAVELENGTH_SWEEP_COUNT)


def x_parallel_coupler_mesh() -> np.ndarray:
    return np.linspace(-PARALLEL_X_HALF_WIDTH_M, PARALLEL_X_HALF_WIDTH_M, PARALLEL_N_X)


def x_parallel_sweep_mesh() -> np.ndarray:
    return np.linspace(-PARALLEL_SWEEP_X_HALF_M, PARALLEL_SWEEP_X_HALF_M, PARALLEL_SWEEP_N_X)


def gap_values_parallel_sweep() -> np.ndarray:
    return np.linspace(PARALLEL_SWEEP_GAP_MIN, PARALLEL_SWEEP_GAP_MAX, PARALLEL_SWEEP_GAP_COUNT)


# --- plot_coupler_heatmap / power / profiles (values that would be library defaults) ---
COUPLER_HEATMAP_UNIT_SCALE = um
COUPLER_HEATMAP_UNIT_LABEL = "um"
COUPLER_HEATMAP_TITLE = "BPM: Gaussian beam in an optical coupler"
COUPLER_HEATMAP_AX = None

COUPLER_POWER_AX = None
COUPLER_POWER_UNIT_SCALE = um
COUPLER_POWER_UNIT_LABEL = "um"
COUPLER_POWER_TITLE = "Power confined to the two output ports"
COUPLER_POWER_UPPER_LABEL = "Upper waveguide"
COUPLER_POWER_LOWER_LABEL = "Lower waveguide"
COUPLER_POWER_TOTAL_LABEL = "Total power"

COUPLER_PROFILES_AX = None
COUPLER_PROFILES_UNIT_SCALE = um
COUPLER_PROFILES_UNIT_LABEL = "um"
COUPLER_PROFILES_TITLE = "Profile comparison"

# plot_coupler_wavelength_response
WAVELENGTH_PLOT_AX = None
WAVELENGTH_PLOT_LAMBDA_REF = COUPLER_LAMBDA_LAUNCH  # reference = lambda_launch from the coupler scenario

# plot_gap_* parallel (keyword defaults)
PARALLEL_GAP_PLOT_CURRENT_GAP = None
PARALLEL_GAP_PLOT_CURRENT_LC = None
PARALLEL_GAP_CURVE_AX = None
PARALLEL_GAP_PEAK_AX = None

# plot_parameter_lc / peak (optional ax)
SWEEP_LC_PLOT_AX = None
SWEEP_PEAK_PLOT_AX = None
SWEEP_PEAK_CURRENT_PARAMETER = None
SWEEP_GAP_PEAK_COUPLER_CURRENT_GAP = None

# simulate_coupler
SIMULATE_COUPLER_LAMBDA_LAUNCH: float | None = None  # uses params.lambda_launch
