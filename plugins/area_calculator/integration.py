from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

_FIT_TO_INTERP = {
    "Linear":       "linear",
    "Cubic Spline": "spline",
    "PCHIP":        "pchip",
    "Poly2":        "poly2",
    "Poly3":        "poly3",
    "Poly4":        "poly4",
}
INTERP_NAMES = list(_FIT_TO_INTERP.keys())
FIT_MODE_KEYS = list(_FIT_TO_INTERP.values())


def get_fit_mode_key(interp_name: str) -> str:
    """Return the pyqt5-chart-widget fit mode key for a display name."""
    return _FIT_TO_INTERP.get(interp_name, "pchip")


def trapz(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.trapz(ys, xs))


def simpson(xs: np.ndarray, ys: np.ndarray) -> float:
    n = len(xs)
    if n < 3:
        return trapz(xs, ys)
    tail = 0.0
    if n % 2 == 0:
        tail = trapz(xs[-2:], ys[-2:])
        xs, ys, n = xs[:-1], ys[:-1], n - 1
    h = (xs[-1] - xs[0]) / (n - 1)
    return tail + h / 3.0 * (
        ys[0]
        + 4.0 * float(np.sum(ys[1:-1:2]))
        + 2.0 * float(np.sum(ys[2:-2:2]))
        + ys[-1]
    )


def gauss_legendre(xs: np.ndarray, ys: np.ndarray) -> float:
    try:
        from scipy.interpolate import PchipInterpolator
        itp = PchipInterpolator(xs, ys)
        a, b = float(xs[0]), float(xs[-1])
        xi, wi = np.polynomial.legendre.leggauss(max(5, len(xs)))
        xm = 0.5 * (b - a) * xi + 0.5 * (a + b)
        return float(0.5 * (b - a) * np.dot(wi, itp(xm)))
    except Exception:
        return trapz(xs, ys)


_METHODS: dict = {
    "Trapezoid":      trapz,
    "Simpson":        simpson,
    "Gauss-Legendre": gauss_legendre,
}
METHOD_NAMES = list(_METHODS.keys())


def _clean(raw_xs: np.ndarray, raw_ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove non-finite values and sort by x."""
    mask = np.isfinite(raw_xs) & np.isfinite(raw_ys)
    xs, ys = raw_xs[mask], raw_ys[mask]
    if len(xs) < 2:
        raise ValueError("Need at least 2 finite data points.")
    order = np.argsort(xs)
    return xs[order], ys[order]


def integrate_from_fit_data(
    fit_xs: List[float],
    fit_ys: List[float],
    method: str,
) -> float:
    """
    Integrate pre-evaluated fit data from a _FitItem.getData() call.
    fit_xs/fit_ys are already dense and regularly sampled by the chart.
    """
    xs = np.asarray(fit_xs, dtype=np.float64)
    ys = np.asarray(fit_ys, dtype=np.float64)
    fin = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[fin], ys[fin]
    if len(xs) < 2:
        raise ValueError("Fit produced no finite values.")
    fn = _METHODS.get(method, trapz)
    return fn(xs, ys)


def integrate_data_raw(
    raw_xs: np.ndarray,
    raw_ys: np.ndarray,
    method: str,
    a: Optional[float],
    b: Optional[float],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Integrate raw tabular data without chart-level interpolation.
    Returns (area, clean_xs, clean_ys).
    """
    xs, ys = _clean(raw_xs, raw_ys)
    lo = max(a, float(xs[0])) if a is not None else float(xs[0])
    hi = min(b, float(xs[-1])) if b is not None else float(xs[-1])
    if lo >= hi:
        raise ValueError(
            f"Range [{lo:.4g}, {hi:.4g}] is empty. "
            f"Data spans [{xs[0]:.4g}, {xs[-1]:.4g}]."
        )
    mask = (xs >= lo - 1e-12) & (xs <= hi + 1e-12)
    xc, yc = xs[mask], ys[mask]
    if len(xc) < 2:
        raise ValueError("No data points in the selected range.")
    fn = _METHODS.get(method, trapz)
    return fn(xc, yc), xs, ys


def integrate_expr(
    expr: str,
    a: float,
    b: float,
    n: int,
    method: str,
    extra: dict,
) -> Tuple[float, np.ndarray, np.ndarray]:
    from worker_proc import _compile, _BASE_NS
    _EMPTY = {"__builtins__": {}}
    xs = np.linspace(a, b, n, dtype=np.float64)
    ns = {**_BASE_NS, "x": xs}
    if extra:
        ns.update(extra)
    raw = eval(_compile(expr), _EMPTY, ns)
    ys = np.asarray(raw, dtype=np.float64)
    if ys.ndim == 0:
        ys = np.full(xs.shape[0], float(ys))
    fin = np.isfinite(ys)
    xs_f, ys_f = xs[fin], ys[fin]
    if len(xs_f) < 2:
        raise ValueError("Expression produced no finite values in [a, b].")
    fn = _METHODS.get(method, trapz)
    return fn(xs_f, ys_f), xs_f, ys_f