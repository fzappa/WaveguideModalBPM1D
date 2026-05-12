"""1D Crank-Nicolson BPM: tridiagonal system without sparse LU factorization."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def crank_nicolson_tridiagonal_coefficients(
    alpha1: complex,
    alpha2: complex,
    t1: float,
    dz: float,
    n_profile: np.ndarray,
    n_eff_ref: float,
    boundary_condition: str = "tbc",
    tbc_gamma: complex | None = None,
):
    """
    Coeficientes de A e B no passo CN: A u^{k+1} = B u^k, com A,B tridiagonais
    (same construction as identity +- h*(dz/2), h = alpha1*t1*L + alpha2*diag(n^2-neff^2)).
    """
    n_prof = np.asarray(n_profile, dtype=np.float64)
    term = n_prof * n_prof - float(n_eff_ref) ** 2
    h_off = alpha1 * t1
    lo_h = np.full(n_prof.size - 1, h_off, dtype=np.complex128)
    mid_h = np.ascontiguousarray(-2.0 * h_off + alpha2 * term, dtype=np.complex128)
    up_h = lo_h.copy()

    if boundary_condition.lower() == "tbc":
        gamma = 0.0 + 0.0j if tbc_gamma is None else complex(tbc_gamma)
        dx = 1.0 / np.sqrt(float(t1))
        # Local first-order DtN: du/dx = gamma*u at left, du/dx = -gamma*u at right.
        mid_h[0] = alpha2 * term[0] + alpha1 * t1 * (-2.0 * (1.0 + dx * gamma))
        up_h[0] = 2.0 * h_off
        lo_h[-1] = 2.0 * h_off
        mid_h[-1] = alpha2 * term[-1] + alpha1 * t1 * (-2.0 * (1.0 + dx * gamma))

    half = 0.5 * dz
    lo_a = -lo_h * half
    mid_a = np.ascontiguousarray(1.0 - mid_h * half, dtype=np.complex128)
    up_a = -up_h * half

    lo_b = lo_h * half
    mid_b = np.ascontiguousarray(1.0 + mid_h * half, dtype=np.complex128)
    up_b = up_h * half

    return lo_a, mid_a, up_a, lo_b, mid_b, up_b


def pack_tridiagonal_banded(
    lo: np.ndarray,
    mid: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """
    Formato SciPy solve_banded com (l, u) = (1, 1): ``ab[u + i - j, j] == a[i, j]``.
    Superdiagonal a[i, i+1] em ab[0, i+1]; diagonal em ab[1, :]; subdiagonal a[i+1, i] em ab[2, i].
    """
    n = mid.shape[0]
    ab = np.zeros((3, n), dtype=np.complex128)
    ab[0, 1:] = up
    ab[1, :] = mid
    ab[2, :-1] = lo
    return ab


def tridiagonal_matvec(
    lo: np.ndarray,
    mid: np.ndarray,
    up: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.complex128)
    y = mid * x
    y[:-1] += up * x[1:]
    y[1:] += lo * x[:-1]
    return y


def solve_banded_tridiagonal(ab: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return solve_banded(
        (1, 1),
        ab,
        np.ascontiguousarray(rhs, dtype=np.complex128),
        overwrite_ab=False,
        overwrite_b=False,
    )
