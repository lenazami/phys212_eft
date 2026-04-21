# ------------------
# Imports
# ------------------
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

# abacusutils -----
try:
    from abacusnbody.metadata import get_meta as _abacus_get_meta
except ImportError:  # not on PATH in IDE/lint environments
    _abacus_get_meta = None  # type: ignore[assignment]


# ------------------
# Functions
# ------------------

def parse_header(path: Path) -> dict:
    """Parse a plain-text Abacus header into a flat key/value dict."""
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"')
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = val
    return out


@lru_cache(maxsize=None)
def get_abacus_metadata(sim_name: str, redshift: float | None = None) -> dict:
    """Fetch official Abacus metadata via abacusutils (cached)."""
    assert _abacus_get_meta is not None, "abacusutils not installed"
    if redshift is None:
        return _abacus_get_meta(sim_name)
    return _abacus_get_meta(sim_name, redshift=float(redshift))


def cosmo_from_header(path: Path) -> dict:
    """Return cosmology dict needed by theory code, sourced from abacusutils.

    Args:
        path: path to a plain-text Abacus header file
    """
    raw       = parse_header(path)
    sim_name  = str(raw["SimName"])
    redshift  = float(raw["Redshift"])

    meta      = get_abacus_metadata(sim_name, redshift)
    omega_b   = float(meta["omega_b"])
    omega_cdm = float(meta["omega_cdm"])
    omega_nu  = float(meta.get("omega_ncdm", 0.0))
    omega_m   = omega_b + omega_cdm + omega_nu
    h0        = float(meta["H0"])
    hh        = h0 / 100.0

    return {
        "sim_name":  sim_name,
        "H0":        h0,
        "h":         hh,
        "omega_b":   omega_b,
        "omega_cdm": omega_cdm,
        "omega_nu":  omega_nu,
        "omega_m":   omega_m,
        "Omega_m":   float(meta.get("Omega_M", omega_m / hh**2)),
        "Omega_b":   omega_b / hh**2,
        "n_s":       float(meta["n_s"]),
        "A_s":       float(meta["A_s"]) if meta.get("A_s") is not None else None,
        "w0":        float(meta.get("w0", -1.0)),
        "wa":        float(meta.get("wa",  0.0)),
        "box_size":  float(meta["BoxSizeHMpc"]),
        "redshift":  redshift,
        "D_z":       float(meta.get("Growth",              1.0)),
        "f_growth":  float(meta.get("f_growth",            1.0)),
        "z_pk":      float(meta.get("ZD_Pk_file_redshift", 1.0)),
    }
