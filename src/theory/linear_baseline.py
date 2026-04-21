# ------------------
# Imports
# ------------------
from __future__ import annotations

import numpy as np
from theory.header_cosmo import get_abacus_metadata

# ------------------
# Functions
# ------------------

def _find_growth_value(growth_table: dict, redshift: float, tol: float = 1e-8) -> float:
    # exact key first, then fuzzy match for floating-point redshifts
    if redshift in growth_table:
        return float(growth_table[redshift])
    for key, val in growth_table.items():
        if abs(float(key) - redshift) <= tol:
            return float(val)
    raise KeyError(f"Redshift {redshift} not in Abacus growth table.")


def _extract_class_power(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    table = meta["CLASS_power_spectrum"]
    k  = np.asarray(table["k (h/Mpc)"],   dtype=np.float64)
    pk = np.asarray(table["P (Mpc/h)^3"], dtype=np.float64)
    return k, pk


def make_linear_pk(k: np.ndarray, cosmo: dict, redshift: float | None = None) -> np.ndarray:
    """Return P_lin(k) in (Mpc/h)^3 using the CLASS spectrum stored in Abacus metadata.

    Args:
        k: wavenumbers in h/Mpc
        cosmo: dict from cosmo_from_header
        redshift: target redshift; defaults to cosmo['redshift']
    """
    target_z = float(cosmo["redshift"] if redshift is None else redshift)
    meta     = get_abacus_metadata(cosmo["sim_name"])

    class_k, class_pk = _extract_class_power(meta)

    # scale CLASS P(k) from its reference redshift to target via growth factors
    growth_table = meta["GrowthTable"]
    z_pk         = float(meta.get("ZD_Pk_file_redshift", 1.0))
    D_target     = _find_growth_value(growth_table, target_z)
    D_pk         = _find_growth_value(growth_table, z_pk)
    scaled_pk    = class_pk * (D_target / D_pk) ** 2

    # log-log interpolation onto k grid
    return np.exp(np.interp(
        np.log(k), np.log(class_k), np.log(scaled_pk),
        left=np.log(scaled_pk[0]), right=np.log(scaled_pk[-1]),
    ))
