# ------------------
# Imports
# ------------------
from __future__ import annotations

# general -----
import numpy as np
from scipy.interpolate import interp1d


# ------------------
# Functions
# ------------------

# SPT kernels -----

def _f2(r: np.ndarray, x: np.ndarray, D: np.ndarray) -> np.ndarray:
    # F2(q, k-q) with k=1, r=q/k, x=cos(k,q), D=|k-q|/k
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = (x - r) * (r**2 + D**2) / (2.0 * r * D**2)
        out = 5.0/7.0 + mid + 2.0*(x - r)**2 / (7.0 * D**2)
    return np.where(D < 1e-10, 0.0, out)


def _f3_integrand(r: np.ndarray, x: np.ndarray,
                  D: np.ndarray, E: np.ndarray) -> np.ndarray:
    # 54 x F3(k,q,-q) in the EdS convention used below.
    # Dividing by 54 gives the kernel that belongs in
    # P13 = 6 P11(k) int d^3q / (2pi)^3 F3(k,q,-q) P11(q).
    def _f2_g2(mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 5.0/7.0 + mid + 2.0*x**2/7.0, 3.0/7.0 + mid + 4.0*x**2/7.0

    mid_kq          = x * (1.0 + r**2) / (2.0 * r)
    F2_kq,  G2_kq  = _f2_g2( mid_kq)
    F2_knq, G2_knq = _f2_g2(-mid_kq)

    safe_D = lambda a: np.where(D < 1e-10, 0.0, a)
    safe_E = lambda a: np.where(E < 1e-10, 0.0, a)

    T1 = safe_D(7.0*(x/r)*F2_knq + (x - r)/(r*D**2)*G2_knq)
    T2 = safe_E(7.0*(-x/r)*F2_kq + (-(x+r))/(r*E**2)*G2_kq)
    T3 = safe_E(G2_kq  * (6.0*r + (7.0*r**2 - 1.0)*x) / (r*E**2))
    T4 = safe_D(G2_knq * (6.0*r + (1.0 - 7.0*r**2)*x) / (r*D**2))
    return T1 + T2 + T3 + T4


# Integration grid -----

def _make_grid(nr: int = 512, nx: int = 256,
               r_min: float = 1e-4, r_max: float = 200.0):
    ln_r  = np.linspace(np.log(r_min), np.log(r_max), nr)
    r     = np.exp(ln_r)
    dr    = np.gradient(ln_r) * r   # d(r) = r d(ln r)
    x, wx = np.polynomial.legendre.leggauss(nx)
    return r, dr, x, wx


# One-loop integrals -----

def compute_p22_p13(
    k_eval: np.ndarray,
    k_lin:  np.ndarray,
    pk_lin: np.ndarray,
    nr:    int   = 512,
    nx:    int   = 128,
    r_min: float = 1e-4,
    r_max: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (P22, P13) at k_eval for a P_lin over k_lin.

    Args:
        k_eval: output wavenumbers in h/Mpc
        k_lin: tabulated linear P(k) wavenumbers
        pk_lin: tabulated linear P(k) values in (Mpc/h)^3
        nr: radial integration points (log-spaced)
        nx: angular integration points (Gauss-Legendre)
    """
    log_pk = interp1d(
        np.log(k_lin), np.log(pk_lin),
        kind="linear", bounds_error=False,
        fill_value=(np.log(pk_lin[0]), -50.0),
    )
    def Plin(k_arr: np.ndarray) -> np.ndarray:
        lk = np.log(np.clip(k_arr, k_lin[0]*0.1, k_lin[-1]*10))
        return np.exp(log_pk(lk))

    r_grid, dr_grid, x_gl, wx_gl = _make_grid(nr, nx, r_min, r_max)
    r  = r_grid[:, None]
    dr = dr_grid[:, None]
    x  = x_gl[None, :]
    wx = wx_gl[None, :]

    D = np.sqrt(np.maximum(1.0 + r**2 - 2.0*r*x, 0.0))
    E = np.sqrt(np.maximum(1.0 + r**2 + 2.0*r*x, 0.0))

    f2_sq = _f2(r, x, D) ** 2
    f3    = _f3_integrand(r, x, D, E) / 54.0

    P22 = np.empty(len(k_eval))
    P13 = np.empty(len(k_eval))
    for i, k in enumerate(k_eval):
        P_r = Plin(k * r)
        P_D = Plin(k * D)
        P_k = Plin(np.array([k]))[0]

        I22 = np.sum(wx * r**2 * f2_sq * P_r * P_D, axis=1)
        I13 = np.sum(wx * r**2 * f3   * P_r,         axis=1)
        P22[i] = 2.0 * k**3 / (4.0*np.pi**2) * np.sum(I22 * dr_grid)
        P13[i] = 6.0 * P_k * k**3 / (4.0*np.pi**2) * np.sum(I13 * dr_grid)

    return P22, P13


def compute_one_loop(
    k_eval: np.ndarray,
    k_lin:  np.ndarray,
    pk_lin: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Returns dict w P11, P22, P13, P_1loop at k_eval."""
    P11 = np.exp(np.interp(np.log(k_eval), np.log(k_lin), np.log(pk_lin)))
    P22, P13 = compute_p22_p13(k_eval, k_lin, pk_lin, **kwargs)
    return {"k": k_eval, "P11": P11, "P22": P22, "P13": P13, "P_1loop": P11 + P22 + P13}
