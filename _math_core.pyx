# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: infer_types=True
# cython: profile=False
# cython: annotation_typing=False

import math
import numpy as np
cimport numpy as cnp
from libc.math cimport isfinite, floor, fabs, cos, sin

cnp.import_array()

ctypedef cnp.float64_t F64
ctypedef cnp.uint8_t U8

DEF _DISC_FACTOR = 10.0
DEF _ADAPTIVE_REFINE_PTS = 128
DEF _ADAPTIVE_MIN_PTS = 16
DEF _ADAPTIVE_OVERSAMPLE = 8

cdef dict _COMPILE_CACHE = {}

cpdef object get_compiled(str expr):
    cdef object c = _COMPILE_CACHE.get(expr)
    if c is None:
        c = compile(expr, "<expr>", "eval")
        _COMPILE_CACHE[expr] = c
    return c

cdef inline cnp.ndarray[F64, ndim=1] _eval_batch_raw(
    object compiled, dict base_ns, cnp.ndarray[F64, ndim=1] x_arr, dict ns_extra
):
    cdef dict ns
    if ns_extra:
        ns = {**base_ns, "x": x_arr}
        ns.update(ns_extra)
    else:
        ns = {**base_ns, "x": x_arr}
    cdef object raw = eval(compiled, {"__builtins__": {}}, ns)
    cdef cnp.ndarray[F64, ndim=1] y = np.asarray(raw, dtype=np.float64)
    if y.ndim == 0:
        y = np.full(x_arr.shape[0], float(y), dtype=np.float64)
    return y

cpdef cnp.ndarray[F64, ndim=1] eval_np_batch(str expr, cnp.ndarray[F64, ndim=1] x_arr, dict base_ns, dict ns_extra):
    """Evaluate expr over x_arr using pre-compiled code and numpy namespace."""
    return _eval_batch_raw(get_compiled(expr), base_ns, x_arr, ns_extra)

cpdef list finalize_y(cnp.ndarray[F64, ndim=1] arr):
    """Convert float64 array to List[Optional[float]], None for non-finite."""
    cdef Py_ssize_t n = arr.shape[0], i
    cdef list out = [None] * n
    cdef double v
    for i in range(n):
        v = arr[i]
        if isfinite(v):
            out[i] = v
    return out

cpdef tuple filter_none_np(object xs, list ys):
    """Filter paired (xs, ys) keeping only finite ys."""
    cdef cnp.ndarray[F64, ndim=1] x_arr = np.asarray(xs, dtype=np.float64)
    cdef Py_ssize_t n = len(ys), i
    cdef cnp.ndarray[F64, ndim=1] y_arr = np.empty(n, dtype=np.float64)
    cdef double nan_val = float("nan")
    cdef object v
    for i in range(n):
        v = ys[i]
        y_arr[i] = <double>v if v is not None else nan_val
    cdef cnp.ndarray[U8, ndim=1, cast=True] mask = np.isfinite(y_arr)
    return x_arr[mask].tolist(), y_arr[mask].tolist()

cpdef list numerical_deriv_cy(
    str expr,
    cnp.ndarray[F64, ndim=1] x_arr,
    dict base_ns,
    dict ns_extra,
    double h
):
    """Central-difference first derivative: (f(x+h) - f(x-h)) / 2h."""
    cdef object c = get_compiled(expr)
    cdef cnp.ndarray[F64, ndim=1] xp = x_arr + h
    cdef cnp.ndarray[F64, ndim=1] xm = x_arr - h
    cdef cnp.ndarray[F64, ndim=1] yp = _eval_batch_raw(c, base_ns, xp, ns_extra)
    cdef cnp.ndarray[F64, ndim=1] ym = _eval_batch_raw(c, base_ns, xm, ns_extra)
    return finalize_y((yp - ym) * (0.5 / h))

cpdef list numerical_deriv2_cy(
    str expr,
    cnp.ndarray[F64, ndim=1] x_arr,
    dict base_ns,
    dict ns_extra,
    double h
):
    """Central-difference second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2."""
    cdef object c = get_compiled(expr)
    cdef cnp.ndarray[F64, ndim=1] yp = _eval_batch_raw(c, base_ns, x_arr + h, ns_extra)
    cdef cnp.ndarray[F64, ndim=1] yc = _eval_batch_raw(c, base_ns, x_arr,     ns_extra)
    cdef cnp.ndarray[F64, ndim=1] ym = _eval_batch_raw(c, base_ns, x_arr - h, ns_extra)
    return finalize_y((yp - 2.0 * yc + ym) * (1.0 / (h * h)))

cpdef list numerical_integral_cy(
    str expr,
    cnp.ndarray[F64, ndim=1] x_arr,
    dict base_ns,
    dict ns_extra
):
    """Cumulative trapezoidal integral."""
    cdef cnp.ndarray[F64, ndim=1] y_arr = _eval_batch_raw(get_compiled(expr), base_ns, x_arr, ns_extra)
    cdef Py_ssize_t n = x_arr.shape[0], i, nm1 = n - 1
    cdef cnp.ndarray[U8, ndim=1, cast=True] fin_mask = np.isfinite(y_arr)
    cdef cnp.ndarray[F64, ndim=1] y_safe = np.where(fin_mask, y_arr, 0.0)
    cdef cnp.ndarray[F64, ndim=1] cumsum = np.empty(n, dtype=np.float64)
    cdef double running = 0.0
    cumsum[0] = 0.0
    for i in range(nm1):
        if fin_mask[i] and fin_mask[i + 1]:
            running += (y_safe[i] + y_safe[i + 1]) * 0.5 * (x_arr[i + 1] - x_arr[i])
        cumsum[i + 1] = running
    cdef list out = [None] * n
    for i in range(n):
        if fin_mask[i]:
            out[i] = cumsum[i]
    return out

cpdef list sample_y_cy(
    str expr,
    cnp.ndarray[F64, ndim=1] x_arr,
    dict base_ns,
    dict ns_extra
):
    """Vectorised y=f(x) sampling with None for non-finite values."""
    return finalize_y(_eval_batch_raw(get_compiled(expr), base_ns, x_arr, ns_extra))

cpdef tuple sample_polar_cy(
    str expr,
    cnp.ndarray[F64, ndim=1] t_arr,
    dict base_ns,
    dict ns_extra
):
    """Vectorised r=f(theta) -> (xs, ys) sampling."""
    cdef dict ns
    if ns_extra:
        ns = {**base_ns, "t": t_arr, "theta": t_arr}
        ns.update(ns_extra)
    else:
        ns = {**base_ns, "t": t_arr, "theta": t_arr}
    cdef cnp.ndarray[F64, ndim=1] r_arr = np.asarray(
        eval(get_compiled(expr), {"__builtins__": {}}, ns), dtype=np.float64
    )
    if r_arr.ndim == 0:
        r_arr = np.full(t_arr.shape[0], float(r_arr), dtype=np.float64)
    cdef cnp.ndarray[U8, ndim=1, cast=True] mask = np.isfinite(r_arr)
    cdef cnp.ndarray[F64, ndim=1] t_f = t_arr[mask]
    cdef cnp.ndarray[F64, ndim=1] r_f = r_arr[mask]
    return (r_f * np.cos(t_f)).tolist(), (r_f * np.sin(t_f)).tolist()

cpdef tuple sample_parametric_cy(
    str x_expr,
    str y_expr,
    cnp.ndarray[F64, ndim=1] t_arr,
    dict base_ns,
    dict ns_extra
):
    """Vectorised parametric x(t), y(t) sampling."""
    cdef dict ns
    if ns_extra:
        ns = {**base_ns, "t": t_arr}
        ns.update(ns_extra)
    else:
        ns = {**base_ns, "t": t_arr}
    cdef Py_ssize_t nt = t_arr.shape[0]
    cdef cnp.ndarray[F64, ndim=1] xv = np.asarray(
        eval(get_compiled(x_expr), {"__builtins__": {}}, ns), dtype=np.float64
    )
    cdef cnp.ndarray[F64, ndim=1] yv = np.asarray(
        eval(get_compiled(y_expr), {"__builtins__": {}}, ns), dtype=np.float64
    )
    if xv.ndim == 0: xv = np.full(nt, float(xv), dtype=np.float64)
    if yv.ndim == 0: yv = np.full(nt, float(yv), dtype=np.float64)
    cdef cnp.ndarray[U8, ndim=1, cast=True] mask = np.isfinite(xv) & np.isfinite(yv)
    return xv[mask].tolist(), yv[mask].tolist()

cdef cnp.ndarray[F64, ndim=1] _np_diff_abs(cnp.ndarray[F64, ndim=1] a, Py_ssize_t n):
    cdef cnp.ndarray[F64, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i
    cdef double diff
    for i in range(n):
        diff = a[i + 1] - a[i]
        out[i] = diff if diff >= 0.0 else -diff
    return out

cdef cnp.ndarray[U8, ndim=1] _both_finite(cnp.ndarray[U8, ndim=1, cast=True] fin, Py_ssize_t n):
    cdef cnp.ndarray[U8, ndim=1] out = np.empty(n, dtype=np.uint8)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = fin[i] & fin[i + 1]
    return out

cpdef tuple sample_y_adaptive_cy(
    str expr,
    cnp.ndarray[F64, ndim=1] x_arr,
    dict base_ns,
    dict ns_extra
):
    """Adaptive y=f(x) sampler with discontinuity detection and local refinement."""
    cdef Py_ssize_t n = x_arr.shape[0]
    if n < 2:
        return list(x_arr), sample_y_cy(expr, x_arr, base_ns, ns_extra)
    cdef object c = get_compiled(expr)
    cdef cnp.ndarray[F64, ndim=1] y_arr
    try:
        y_arr = _eval_batch_raw(c, base_ns, x_arr, ns_extra)
    except Exception:
        return list(x_arr), [None] * n
    cdef cnp.ndarray[U8, ndim=1, cast=True] fin = np.isfinite(y_arr)
    cdef cnp.ndarray[F64, ndim=1] y_work = np.where(fin, y_arr, np.nan)
    cdef cnp.ndarray[F64, ndim=1] fin_vals = y_arr[fin]
    cdef double y_range = float(np.ptp(fin_vals)) if len(fin_vals) >= 2 else 1.0
    if y_range == 0.0:
        y_range = 1.0
    cdef double threshold = y_range * _DISC_FACTOR
    cdef Py_ssize_t nm1 = n - 1
    cdef cnp.ndarray[F64, ndim=1] dy_abs = _np_diff_abs(y_work, nm1)
    cdef cnp.ndarray[U8, ndim=1] both_fin = _both_finite(fin, nm1)
    cdef cnp.ndarray jumps = np.where((dy_abs > threshold) | (both_fin == 0))[0]
    cdef Py_ssize_t nj = len(jumps), ji
    for ji in jumps:
        y_work[ji] = np.nan
        if ji + 1 < n:
            y_work[ji + 1] = np.nan
    if nj == 0:
        return list(x_arr), finalize_y(y_work)
    cdef double dx_step = float(x_arr[1] - x_arr[0])
    cdef list ranges = []
    cdef double xl, xr
    cdef Py_ssize_t lo_idx, hi_idx
    for ji in jumps:
        lo_idx = ji - 1 if ji > 0 else 0
        hi_idx = ji + 2 if ji + 2 < n else nm1
        ranges.append((float(x_arr[lo_idx]), float(x_arr[hi_idx])))
    cdef list merged = []
    cdef Py_ssize_t nm_len
    for rng in sorted(ranges):
        nm_len = len(merged)
        if nm_len > 0 and rng[0] <= merged[nm_len - 1][1] + dx_step * 2:
            if rng[1] > merged[nm_len - 1][1]:
                merged[nm_len - 1][1] = rng[1]
        else:
            merged.append([rng[0], rng[1]])
    cdef list sub_arrs = []
    cdef int pts
    cdef Py_ssize_t nm = len(merged), i
    for i in range(nm):
        xl = merged[i][0]
        xr = merged[i][1]
        pts = int((xr - xl) / (dx_step if dx_step > 1e-30 else 1e-30) * _ADAPTIVE_OVERSAMPLE)
        if pts < _ADAPTIVE_MIN_PTS: pts = _ADAPTIVE_MIN_PTS
        if pts > _ADAPTIVE_REFINE_PTS: pts = _ADAPTIVE_REFINE_PTS
        sub_arrs.append(np.linspace(xl, xr, pts, dtype=np.float64))
    cdef cnp.ndarray[F64, ndim=1] all_rx, all_ry
    all_rx = np.concatenate(sub_arrs) if nm > 0 else np.empty(0, dtype=np.float64)
    cdef double ry_range, ry_thresh
    cdef Py_ssize_t nrx = all_rx.shape[0], di, nrxm1
    cdef cnp.ndarray[F64, ndim=1] dy_r_arr
    cdef cnp.ndarray[U8, ndim=1] both_r_arr
    if nrx > 0:
        try:
            all_ry = _eval_batch_raw(c, base_ns, all_rx, ns_extra)
            ry_fin_mask = np.isfinite(all_ry)
            ry_vals = all_ry[ry_fin_mask]
            ry_range = float(np.ptp(ry_vals)) if len(ry_vals) else y_range
            ry_thresh = (ry_range or y_range) * _DISC_FACTOR
            if threshold > ry_thresh:
                ry_thresh = threshold
            nrxm1 = nrx - 1
            dy_r_arr = _np_diff_abs(all_ry, nrxm1)
            both_r_arr = _both_finite(np.asarray(ry_fin_mask, dtype=np.uint8), nrxm1)
            disc_r = np.where((dy_r_arr > ry_thresh) | (both_r_arr == 0))[0]
            for di in disc_r:
                all_ry[di] = np.nan
                if di + 1 < nrx:
                    all_ry[di + 1] = np.nan
        except Exception:
            all_rx = np.empty(0, dtype=np.float64)
            all_ry = np.empty(0, dtype=np.float64)
    else:
        all_ry = np.empty(0, dtype=np.float64)
    nrx = all_rx.shape[0]
    cdef cnp.ndarray[F64, ndim=1] combined_x, combined_y
    if nm > 0 and nrx > 0:
        covered = np.zeros(n, dtype=bool)
        for i in range(nm):
            covered |= (x_arr >= merged[i][0]) & (x_arr <= merged[i][1])
        keep = ~covered
        combined_x = np.concatenate([x_arr[keep], all_rx])
        combined_y = np.concatenate([y_work[keep], all_ry])
        sort_idx = np.argsort(combined_x, kind="stable")
        combined_x = combined_x[sort_idx]
        combined_y = combined_y[sort_idx]
    else:
        combined_x = x_arr
        combined_y = y_work
    return combined_x.tolist(), finalize_y(np.ascontiguousarray(combined_y, dtype=np.float64))

cpdef cnp.ndarray[F64, ndim=1] linspace_cy(double a, double b, Py_ssize_t n):
    """Fast linspace returning a numpy array."""
    return np.linspace(a, b, n, dtype=np.float64)

cpdef list batch_sample_y(list exprs, cnp.ndarray[F64, ndim=1] x_arr, dict base_ns, dict ns_extra):
    """Evaluate multiple expressions over the same x_arr, returning list of result lists."""
    cdef list results = []
    cdef str expr
    for expr in exprs:
        try:
            results.append(finalize_y(_eval_batch_raw(get_compiled(expr), base_ns, x_arr, ns_extra)))
        except Exception:
            results.append([None] * x_arr.shape[0])
    return results

cpdef tuple batch_deriv_trio(
    str expr,
    cnp.ndarray[F64, ndim=1] x_arr,
    dict base_ns,
    dict ns_extra,
    double h,
    bint want_d1,
    bint want_d2,
    bint want_int
):
    """Compute d1, d2, integral in one pass, reusing shared evaluations."""
    cdef object c = get_compiled(expr)
    cdef cnp.ndarray[F64, ndim=1] yc, yp, ym
    cdef list d1 = [], d2 = [], ig = []
    cdef Py_ssize_t n = x_arr.shape[0], i, nm1 = n - 1
    cdef double running
    cdef cnp.ndarray[U8, ndim=1, cast=True] fin_mask
    if want_d1 or want_d2:
        yp = _eval_batch_raw(c, base_ns, x_arr + h, ns_extra)
        ym = _eval_batch_raw(c, base_ns, x_arr - h, ns_extra)
        if want_d1:
            d1 = finalize_y((yp - ym) * (0.5 / h))
        if want_d2:
            yc = _eval_batch_raw(c, base_ns, x_arr, ns_extra)
            d2 = finalize_y((yp - 2.0 * yc + ym) * (1.0 / (h * h)))
    if want_int:
        if not want_d2:
            yc = _eval_batch_raw(c, base_ns, x_arr, ns_extra)
        fin_mask = np.isfinite(yc)
        y_safe = np.where(fin_mask, yc, 0.0)
        running = 0.0
        ig = [None] * n
        ig[0] = 0.0 if fin_mask[0] else None
        for i in range(nm1):
            if fin_mask[i] and fin_mask[i + 1]:
                running += (y_safe[i] + y_safe[i + 1]) * 0.5 * (x_arr[i + 1] - x_arr[i])
            if fin_mask[i + 1]:
                ig[i + 1] = running
    return d1, d2, ig
