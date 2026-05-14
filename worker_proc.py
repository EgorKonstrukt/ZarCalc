"""
Worker process that lives in a separate Python interpreter.
Receives tasks via multiprocessing.Queue, returns results via result Queue.
Each worker builds its own namespace once and reuses it.
"""
from __future__ import annotations
import math
import traceback
from multiprocessing import Queue

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

_COMPILE_CACHE: dict = {}

_BASE_NS: dict = {}
if _NP_OK:
    _BASE_NS = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
        "log2": np.log2, "log10": np.log10,
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
        "round": np.round, "pi": math.pi, "e": math.e, "inf": math.inf,
        "sign": np.sign,
        "frac": lambda x: x - np.floor(x),
        "clamp": lambda x, a, b: np.clip(x, a, b),
        "mod": np.fmod,
        "hypot": np.hypot,
        "factorial": math.factorial,
        "degrees": np.degrees,
        "radians": np.radians,
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "step": lambda x: np.where(x >= 0, 1.0, 0.0),
        "rect": lambda x: np.where(np.abs(x) <= 0.5, 1.0, 0.0),
        "tri": lambda x: np.maximum(0.0, 1.0 - np.abs(x)),
        "sawtooth": lambda x: 2.0 * (x / (2 * math.pi) - np.floor(0.5 + x / (2 * math.pi))),
        "square": lambda x: np.where(np.sin(x) >= 0, 1.0, -1.0),
        "sinc": lambda x: np.where(x != 0, np.sin(math.pi * x) / (math.pi * x), 1.0),
        "gaussian": lambda x: np.exp(-x * x / 2.0),
        "lerp": lambda a, b, t: a + (b - a) * t,
        "__builtins__": {},
    }

_EMPTY_BUILTINS = {"__builtins__": {}}


def _compile(expr: str) -> object:
    c = _COMPILE_CACHE.get(expr)
    if c is None:
        c = compile(expr, "<expr>", "eval")
        _COMPILE_CACHE[expr] = c
    return c


def _eval_arr(expr: str, x_arr, extra: dict):
    ns = {**_BASE_NS, "x": x_arr}
    if extra:
        ns.update(extra)
    raw = eval(_compile(expr), _EMPTY_BUILTINS, ns)
    y = np.asarray(raw, dtype=np.float64)
    if y.ndim == 0:
        y = np.full(x_arr.shape[0], float(y), dtype=np.float64)
    return y


def _handle_yfx(task_id: str, expr: str, x_arr, extra: dict):
    y = _eval_arr(expr, x_arr, extra)
    fin = np.isfinite(y)
    return task_id, (x_arr[fin].tolist(), y[fin].tolist())


def _handle_polar(task_id: str, expr: str, x_arr, extra: dict):
    t_arr = np.linspace(0.0, 2 * math.pi, x_arr.shape[0], dtype=np.float64)
    ns = {**_BASE_NS, "t": t_arr, "theta": t_arr}
    if extra:
        ns.update(extra)
    r_arr = np.asarray(eval(_compile(expr), _EMPTY_BUILTINS, ns), dtype=np.float64)
    if r_arr.ndim == 0:
        r_arr = np.full(t_arr.shape[0], float(r_arr), dtype=np.float64)
    mask = np.isfinite(r_arr)
    t_f, r_f = t_arr[mask], r_arr[mask]
    return task_id, ((r_f * np.cos(t_f)).tolist(), (r_f * np.sin(t_f)).tolist())


def _handle_param(task_id: str, expr: str, expr2: str, t_arr, extra: dict):
    ns = {**_BASE_NS, "t": t_arr}
    if extra:
        ns.update(extra)
    xv = np.asarray(eval(_compile(expr), _EMPTY_BUILTINS, ns), dtype=np.float64)
    yv = np.asarray(eval(_compile(expr2), _EMPTY_BUILTINS, ns), dtype=np.float64)
    if xv.ndim == 0: xv = np.full(t_arr.shape[0], float(xv), dtype=np.float64)
    if yv.ndim == 0: yv = np.full(t_arr.shape[0], float(yv), dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    return task_id, (xv[mask].tolist(), yv[mask].tolist())


def _handle_deriv(task_id: str, expr: str, x_arr, extra: dict, h: float):
    xp, xm = x_arr + h, x_arr - h
    c = _compile(expr)
    ns_p = {**_BASE_NS, "x": xp}
    ns_m = {**_BASE_NS, "x": xm}
    if extra:
        ns_p.update(extra)
        ns_m.update(extra)
    yp = np.asarray(eval(c, _EMPTY_BUILTINS, ns_p), dtype=np.float64)
    ym = np.asarray(eval(c, _EMPTY_BUILTINS, ns_m), dtype=np.float64)
    if yp.ndim == 0: yp = np.full(x_arr.shape[0], float(yp), dtype=np.float64)
    if ym.ndim == 0: ym = np.full(x_arr.shape[0], float(ym), dtype=np.float64)
    d = (yp - ym) * (0.5 / h)
    fin = np.isfinite(d)
    return task_id, (x_arr[fin].tolist(), d[fin].tolist())


def _handle_deriv2(task_id: str, expr: str, x_arr, extra: dict, h: float):
    c = _compile(expr)
    def _ev(xa):
        ns = {**_BASE_NS, "x": xa}
        if extra: ns.update(extra)
        r = np.asarray(eval(c, _EMPTY_BUILTINS, ns), dtype=np.float64)
        if r.ndim == 0: r = np.full(x_arr.shape[0], float(r), dtype=np.float64)
        return r
    yp, yc, ym = _ev(x_arr + h), _ev(x_arr), _ev(x_arr - h)
    d2 = (yp - 2.0 * yc + ym) / (h * h)
    fin = np.isfinite(d2)
    return task_id, (x_arr[fin].tolist(), d2[fin].tolist())


def _handle_integral(task_id: str, expr: str, x_arr, extra: dict):
    y = _eval_arr(expr, x_arr, extra)
    fin = np.isfinite(y)
    y_safe = np.where(fin, y, 0.0)
    n = x_arr.shape[0]
    cumsum = np.empty(n, dtype=np.float64)
    running = 0.0
    cumsum[0] = 0.0
    for i in range(n - 1):
        if fin[i] and fin[i + 1]:
            running += (y_safe[i] + y_safe[i + 1]) * 0.5 * (x_arr[i + 1] - x_arr[i])
        cumsum[i + 1] = running
    valid = fin
    return task_id, (x_arr[valid].tolist(), cumsum[valid].tolist())


_HANDLERS = {
    "yfx":      _handle_yfx,
    "polar":    _handle_polar,
    "param":    _handle_param,
    "deriv":    _handle_deriv,
    "deriv2":   _handle_deriv2,
    "integral": _handle_integral,
}


def worker_loop(task_q: Queue, result_q: Queue) -> None:
    """Main loop of a worker process. Reads tasks, computes, sends results."""
    while True:
        try:
            msg = task_q.get()
            if msg is None:
                break
            kind = msg[0]
            handler = _HANDLERS.get(kind)
            if handler is None:
                continue
            try:
                result = handler(*msg[1:])
                result_q.put(("ok", result))
            except Exception as exc:
                result_q.put(("err", (msg[1], ([], []))))
        except Exception:
            pass