import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from WaveguideModalBPM1D import NotebookConfig, WaveguideModalBPM1D, um
from WaveguideModalBPM1D.bpm_tridiagonal import (
    crank_nicolson_tridiagonal_coefficients,
    pack_tridiagonal_banded,
    solve_banded_tridiagonal,
    tridiagonal_matvec,
)
from WaveguideModalBPM1D.parameters import CouplerTestParams, ModalStructureParams


def test_import_and_alias():
    config = NotebookConfig()
    analyzer = WaveguideModalBPM1D(config)

    assert analyzer.lambda_0 == config.lambda_0


def test_modal_structure_params_validation():
    params = ModalStructureParams()

    assert len(params.refractive_indices) == len(params.thicknesses)
    assert params.n_core >= params.n_clad


def test_coupler_params_geometry():
    params = CouplerTestParams(length_total=1764.0 * um, length_coupling=882.0 * um)

    assert params.bend_length > 0.0


def test_solve_modes_runs():
    config = NotebookConfig(n_points=200)
    analyzer = WaveguideModalBPM1D(config)

    result = analyzer.solve_modes(polarization="TE", k_search=4)

    assert "n_effs" in result
    assert result["x"].shape[0] == 200


def test_bpm_tridiagonal_step_matches_sparse_lu():
    """CN step: same solution as splu over CSR (numerical reference)."""
    np.random.seed(0)
    n = 120
    x = np.linspace(-1e-6, 1e-6, n)
    n_profile = 3.5 + 0.05 * np.sin(np.linspace(0, 3, n))
    alpha1 = -1j / 2.5e6
    alpha2 = -1j * 2.5e6
    t1 = 1.0 / (x[1] - x[0]) ** 2
    dz = 0.4e-6
    neff = 3.48

    lap = diags([np.ones(n - 1), np.full(n, -2.0), np.ones(n - 1)], [-1, 0, 1], format="csr")
    identity = diags([np.ones(n)], [0], format="csr")
    term = diags([n_profile**2 - neff**2], [0], format="csr")
    h = alpha1 * t1 * lap + alpha2 * term
    mat_a = identity - h * (dz / 2)
    mat_b = identity + h * (dz / 2)

    u = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex128)
    rhs_sparse = mat_b @ u
    expected = splu(mat_a.tocsc()).solve(rhs_sparse)

    lo_a, mid_a, up_a, lo_b, mid_b, up_b = crank_nicolson_tridiagonal_coefficients(
        alpha1, alpha2, t1, dz, n_profile, neff
    )
    ab = pack_tridiagonal_banded(lo_a, mid_a, up_a)
    rhs_band = tridiagonal_matvec(lo_b, mid_b, up_b, u)
    out = solve_banded_tridiagonal(ab, rhs_band)

    rel = np.linalg.norm(expected - out) / np.linalg.norm(expected)
    assert rel < 1e-12
