# ------------------
# Imports
# ------------------
from __future__ import annotations

# general -----
import numpy as np


# ------------------
# Functions
# ------------------
def evaluate_eft_model(
    k: np.ndarray | None = None,
    p11: np.ndarray | None = None,
    p22: np.ndarray | None = None,
    p13: np.ndarray | None = None,
    c: float | None = None,
    **legacy_kwargs,
) -> dict[str, np.ndarray]:
    """Return all EFT spectrum components at given k.

    Legacy notebook/test keyword names are still accepted for convenience:
    `k_h_per_Mpc`, `p11_Mpc3_per_h3`, `p22_Mpc3_per_h3`, `p13_Mpc3_per_h3`,
    and `c_counterterm_Mpch2`.
    """
    if k is None:
        k = legacy_kwargs.pop("k_h_per_Mpc", None)
    if p11 is None:
        p11 = legacy_kwargs.pop("p11_Mpc3_per_h3", None)
    if p22 is None:
        p22 = legacy_kwargs.pop("p22_Mpc3_per_h3", None)
    if p13 is None:
        p13 = legacy_kwargs.pop("p13_Mpc3_per_h3", None)
    if c is None:
        c = legacy_kwargs.pop("c_counterterm_Mpch2", None)

    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unknown}")

    missing = [
        name for name, value in {
            "k": k,
            "p11": p11,
            "p22": p22,
            "p13": p13,
            "c": c,
        }.items()
        if value is None
    ]
    if missing:
        raise TypeError(f"Missing required EFT inputs: {', '.join(missing)}")

    k = np.asarray(k, dtype=np.float64)
    p11 = np.asarray(p11, dtype=np.float64)
    p22 = np.asarray(p22, dtype=np.float64)
    p13 = np.asarray(p13, dtype=np.float64)
    c = float(c)

    p_1loop = p11 + p22 + p13
    p_ctr   = -2.0 * c * k**2 * p11
    return {
        "k":       k,
        "P11":     p11,
        "P22":     p22,
        "P13":     p13,
        "P_1loop": p_1loop,
        "P_ctr":   p_ctr,
        "P_eft":   p_1loop + p_ctr,
    }
