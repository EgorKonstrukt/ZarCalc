from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen

from pyqt5_chart_widget import ChartWidget, _FunctionItem
from math_engine import (
    linspace, _eval_np_batch, _finalize_y, _NP_OK, _NP_NS,
)
try:
    import _math_core as _cy
    _CY_OK = True
except ImportError:
    _CY_OK = False
from config import Config
from compute_pool import get_pool
from constants import DERIV_H

if TYPE_CHECKING:
    from core.panels import FunctionPanel

_POLAR_THETA_START = 0.0
_POLAR_THETA_END = 2 * math.pi
_EMPTY_XY: Tuple[List, List] = ([], [])
_DERIV_COLORS = {"_d": "#9b59b6", "_d2": "#e74c3c", "_int": "#2ecc71"}
_DERIV_LABELS = {"_d": "f'(x)", "_d2": "f''(x)", "_int": "integral f dx"}
_DRAIN_INTERVAL_MS = 8


class _LiveFn:
    """Mutable callable backing a _FunctionItem; updates in-place."""
    __slots__ = ("_expr", "_extra")

    def __init__(self):
        self._expr: str = ""
        self._extra: dict = {}

    def update(self, expr: str, extra: dict) -> bool:
        changed = (expr != self._expr) or (extra != self._extra)
        if changed:
            self._expr = expr
            self._extra = dict(extra)
        return changed

    def __call__(self, xs: List[float]) -> List[Optional[float]]:
        if not self._expr:
            return [None] * len(xs)
        try:
            x_arr = np.ascontiguousarray(xs, dtype=np.float64)
            if _CY_OK:
                return _cy.sample_y_cy(self._expr, x_arr, _NP_NS, self._extra)
            return _finalize_y(_eval_np_batch(self._expr, x_arr, self._extra))
        except Exception:
            return [None] * len(xs)


class _RowState:
    """Bundles a _FunctionItem and its _LiveFn for one function row."""
    __slots__ = ("item", "live_fn")

    def __init__(self, item: _FunctionItem, live_fn: _LiveFn):
        self.item = item
        self.live_fn = live_fn


class _InflightTask:
    """Metadata kept while a task is being computed in a worker process."""
    __slots__ = ("task_id", "row_ref", "mode", "label", "color", "width", "enabled")

    def __init__(self, task_id, row_ref, mode, label, color, width, enabled):
        self.task_id = task_id
        self.row_ref = row_ref
        self.mode = mode
        self.label = label
        self.color = color
        self.width = width
        self.enabled = enabled


class _InflightDeriv:
    """Metadata for a derivative/integral overlay task."""
    __slots__ = ("task_id", "line_key", "dl_ref")

    def __init__(self, task_id, line_key, dl_ref):
        self.task_id = task_id
        self.line_key = line_key
        self.dl_ref = dl_ref


class Plotter:
    """Evaluates all function rows via worker processes and flushes results to ChartWidget."""

    def __init__(self, chart: ChartWidget, panel: "FunctionPanel"):
        self._chart = chart
        self._panel = panel
        self._cfg = Config()
        self._row_states: Dict[int, _RowState] = {}
        self._pool = get_pool()
        self._inflight_rows: Dict[str, _InflightTask] = {}
        self._inflight_derivs: Dict[str, _InflightDeriv] = {}
        self._drain_timer = QTimer()
        self._drain_timer.setSingleShot(False)
        self._drain_timer.setInterval(_DRAIN_INTERVAL_MS)
        self._drain_timer.timeout.connect(self._drain)

    def set_animating(self, val: bool):
        pass

    def _resolution(self) -> float:
        return getattr(self._cfg, "fn_resolution", 1.5)

    def _get_or_create_state(self, row_id: int, row) -> _RowState:
        if row_id not in self._row_states:
            live_fn = _LiveFn()
            pen = QPen(QColor(row.color), row.get_width())
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            item = _FunctionItem(
                self._chart, live_fn, pen, label="", resolution=self._resolution(),
            )
            self._chart._functions.append(item)
            self._row_states[row_id] = _RowState(item, live_fn)
        return self._row_states[row_id]

    def _remove_state(self, row_id: int):
        state = self._row_states.pop(row_id, None)
        if state is not None:
            self._chart.removeItem(state.item)

    def sync_fn_items(self):
        live_ids = {id(row) for row in self._panel.func_rows}
        for rid in list(self._row_states):
            if rid not in live_ids:
                self._remove_state(rid)

    def _drain(self):
        results = self._pool.drain_results()
        if not results and not self._inflight_rows and not self._inflight_derivs:
            self._drain_timer.stop()
            return
        for task_id, (xs, ys) in results:
            row_task = self._inflight_rows.pop(task_id, None)
            if row_task is not None:
                line = getattr(row_task.row_ref, "chart_line", None)
                if line is not None:
                    suffix = "" if row_task.mode == "y=f(x)" else (" (r)" if row_task.mode == "r=f(t)" else " (p)")
                    line.setLabel(row_task.label + suffix)
                    line.pen.setColor(QColor(row_task.color))
                    line.pen.setWidth(row_task.width)
                    line.setData(xs=xs, ys=ys)
                    line.setVisible(row_task.enabled)
                continue
            deriv_task = self._inflight_derivs.pop(task_id, None)
            if deriv_task is not None:
                dl = deriv_task.dl_ref
                k = deriv_task.line_key
                if k in dl:
                    dl[k].setData(xs=xs, ys=ys)
                    dl[k].setVisible(True)

    def replot(self):
        s = self._panel.settings
        x_min, x_max = s.xmin(), s.xmax()
        if x_min >= x_max:
            return None
        any_anim = any(w._animating for w in self._panel._param_widgets.values())
        n = self._cfg.anim_samples if any_anim else s.samples()
        t_min, t_max = s.tmin(), s.tmax()
        infinite = s.infinite()
        x_arr = np.ascontiguousarray(linspace(x_min, x_max, n), dtype=np.float64)
        t_arr = np.ascontiguousarray(linspace(t_min, t_max, n), dtype=np.float64)
        extra = self._panel.get_params()
        self.sync_fn_items()
        total = 0
        for i, row in enumerate(self._panel.func_rows):
            expr = row.get_expr()
            mode = row.get_mode()
            label = f"f{i + 1}"
            row_id = id(row)
            if infinite and mode == "y=f(x)":
                state = self._get_or_create_state(row_id, row)
                if state.live_fn.update(expr, extra):
                    state.item.invalidateCache()
                    state.item._expr = expr
                    state.item._extra = dict(extra)
                    state.item._adaptive = True
                state.item.label = label
                state.item.pen.setColor(QColor(row.color))
                state.item.pen.setWidth(row.get_width())
                state.item.setVisible(row.is_enabled())
                if row.chart_line is not None:
                    row.chart_line.setData(xs=[], ys=[])
                continue
            self._remove_state(row_id)
            line = row.chart_line
            if line is None:
                continue
            if not expr:
                line.setData(xs=[], ys=[])
                continue
            task_id = str(uuid.uuid4())
            meta = _InflightTask(
                task_id, row, mode, label,
                row.color, row.get_width(), row.is_enabled(),
            )
            self._inflight_rows[task_id] = meta
            if mode == "y=f(x)":
                self._pool.submit_yfx(task_id, expr, x_arr, dict(extra))
            elif mode == "r=f(t)":
                self._pool.submit_polar(task_id, expr, x_arr, dict(extra))
            elif mode == "param":
                expr2 = row.get_expr2()
                if not expr2:
                    self._inflight_rows.pop(task_id, None)
                    line.setData(xs=[], ys=[])
                    continue
                self._pool.submit_param(task_id, expr, expr2, t_arr, dict(extra))
            total += n
        self._submit_derivs(x_arr, extra)
        if not self._drain_timer.isActive():
            self._drain_timer.start()
        return total, x_min, x_max, len(self._panel.func_rows)

    def _submit_derivs(self, x_arr, extra: dict):
        dp = self._panel.deriv_panel
        idx = dp.source_idx()
        rows = self._panel.func_rows
        dl = self._panel.get_deriv_lines()
        active_keys: set = set()
        if 0 <= idx < len(rows):
            row = rows[idx]
            expr = row.get_expr()
            if row.get_mode() == "y=f(x)" and expr:
                shows = {"_d": dp.show_d1(), "_d2": dp.show_d2(), "_int": dp.show_ig()}
                rid = id(row)
                for sfx, visible in shows.items():
                    if not visible:
                        continue
                    k = f"{rid}{sfx}"
                    active_keys.add(k)
                    if k not in dl:
                        dl[k] = self._chart.plot(
                            label=_DERIV_LABELS[sfx],
                            color=_DERIV_COLORS[sfx],
                            width=1,
                        )
                    task_id = str(uuid.uuid4())
                    self._inflight_derivs[task_id] = _InflightDeriv(task_id, k, dl)
                    if sfx == "_d":
                        self._pool.submit_deriv(task_id, expr, x_arr, dict(extra), DERIV_H)
                    elif sfx == "_d2":
                        self._pool.submit_deriv2(task_id, expr, x_arr, dict(extra), DERIV_H)
                    else:
                        self._pool.submit_integral(task_id, expr, x_arr, dict(extra))
        for k in list(dl.keys()):
            if k not in active_keys:
                dl[k].setData(xs=[], ys=[])