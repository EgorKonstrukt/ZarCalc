from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QLocale, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from .data_table import DataTable
from .integration import (
    METHOD_NAMES, INTERP_NAMES, FIT_MODE_KEYS,
    get_fit_mode_key,
    integrate_expr, integrate_data_raw, integrate_from_fit_data,
)
from .chart_overlay import ChartOverlay, _FIT_N_PTS

if TYPE_CHECKING:
    from core.plugins.app_context import AppContext

_DEFAULT_N_EXPR   = 1000
_DEFAULT_N_FIT    = _FIT_N_PTS
_DEFAULT_INTERP   = "PCHIP"
_DEFAULT_METHOD   = "Trapezoid"
_SPIN_EXPR_RANGE  = (4, 100_000)
_SPIN_FIT_RANGE   = (50, 50_000)
_BOUNDS_RANGE     = (-1e9, 1e9)


def _lbl(text: str, bold: bool = False, color: str = "") -> QLabel:
    w = QLabel(text)
    st = ""
    if bold:
        st += "font-weight:bold;"
    if color:
        st += f"color:{color};"
    if st:
        w.setStyleSheet(st)
    return w


def _dspin(lo: float, hi: float, val: float, dec: int = 6, step: float = 0.1) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setLocale(QLocale(QLocale.C))
    return s


def _btn(label: str, bg: str, hov: str) -> QPushButton:
    b = QPushButton(label)
    b.setCursor(Qt.PointingHandCursor)
    return b


class _ResultBox(QWidget):
    """Displays the computed area with metadata or an error message."""
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        self._val  = _lbl("—", bold=True)
        self._meta = _lbl("", color="#6b7280")
        self._meta.setStyleSheet("font-size:10px;color:#6b7280;")
        lay.addWidget(_lbl("Area", bold=True, color="#15803d"))
        lay.addWidget(self._val)
        lay.addWidget(self._meta)

    def show_result(self, area: float, lo: float, hi: float, n: int, method: str, interp: str = "") -> None:
        self._val.setText(f"{area:.8g}")
        interp_str = f"  interp={interp}" if interp else ""
        self._meta.setText(f"[{lo:.4g}, {hi:.4g}]  n={n}  {method}{interp_str}")

    def show_error(self, msg: str) -> None:
        self._val.setText("Error")
        self._meta.setText(msg)

    def clear(self) -> None:
        self._val.setText("—")
        self._meta.setText("")


class _ExprTab(QWidget):
    """Tab for integrating an analytic expression f(x)."""
    compute_requested = pyqtSignal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        g = QGroupBox("Expression  y = f(x)")
        gl = QVBoxLayout(g)
        gl.setSpacing(6)
        row1 = QHBoxLayout()
        row1.addWidget(_lbl("f(x) ="))
        self._expr = QLineEdit()
        self._expr.setPlaceholderText("sin(x)  or  x**2  etc.")
        self._expr.returnPressed.connect(self.compute_requested.emit)
        row1.addWidget(self._expr, 1)
        gl.addLayout(row1)
        row2 = QHBoxLayout()
        for lbl, attr, default in (("a =", "_a", 0.0), ("b =", "_b", 1.0)):
            row2.addWidget(_lbl(lbl))
            spin = _dspin(*_BOUNDS_RANGE, default)
            setattr(self, attr, spin)
            row2.addWidget(spin)
            row2.addSpacing(8)
        row2.addWidget(_lbl("n ="))
        self._n = QSpinBox()
        self._n.setRange(*_SPIN_EXPR_RANGE)
        self._n.setValue(_DEFAULT_N_EXPR)
        self._n.setSingleStep(100)
        row2.addWidget(self._n)
        row2.addStretch()
        gl.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(_lbl("Method:"))
        self._method = QComboBox()
        self._method.addItems(METHOD_NAMES)
        row3.addWidget(self._method)
        row3.addStretch()
        gl.addLayout(row3)
        lay.addWidget(g)
        pg = QGroupBox("Extra parameters  (name = value, one per line)")
        pl = QVBoxLayout(pg)
        self._params = QTextEdit()
        self._params.setPlaceholderText("y_star = 0.5\nk = 2")
        self._params.setFixedHeight(64)
        self._params.setStyleSheet("font-size:11px;")
        pl.addWidget(self._params)
        lay.addWidget(pg)
        lay.addStretch()

    def _parse_params(self) -> dict:
        out = {}
        for line in self._params.toPlainText().splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            try:
                out[k.strip()] = float(v.strip().replace(",", "."))
            except ValueError:
                pass
        return out

    def compute(self) -> Tuple[float, float, float, int, str, np.ndarray, np.ndarray]:
        expr = self._expr.text().strip()
        if not expr:
            raise ValueError("Enter an expression f(x).")
        a, b, n = self._a.value(), self._b.value(), self._n.value()
        method = self._method.currentText()
        area, xs, ys = integrate_expr(expr, a, b, n, method, self._parse_params())
        return area, a, b, n, method, xs, ys

    def to_state(self) -> dict:
        return {
            "expr":   self._expr.text(),
            "a":      self._a.value(),
            "b":      self._b.value(),
            "n":      self._n.value(),
            "method": self._method.currentText(),
            "params": self._params.toPlainText(),
        }

    def apply_state(self, s: dict) -> None:
        self._expr.setText(s.get("expr", ""))
        self._a.setValue(s.get("a", 0.0))
        self._b.setValue(s.get("b", 1.0))
        self._n.setValue(s.get("n", _DEFAULT_N_EXPR))
        idx = self._method.findText(s.get("method", _DEFAULT_METHOD))
        if idx >= 0:
            self._method.setCurrentIndex(idx)
        self._params.setPlainText(s.get("params", ""))


class _DataTab(QWidget):
    """Tab for integrating tabular (x, y) data using chart-native fit interpolation."""
    compute_requested = pyqtSignal()
    interp_changed    = pyqtSignal(str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        self._table = DataTable()
        lay.addWidget(self._table, 1)
        og = QGroupBox("Integration options")
        ol = QVBoxLayout(og)
        ol.setSpacing(6)
        brow = QHBoxLayout()
        self._use_bounds = QCheckBox("Restrict to [a, b]")
        self._use_bounds.stateChanged.connect(self._toggle_bounds)
        brow.addWidget(self._use_bounds)
        brow.addSpacing(8)
        self._a_lbl = _lbl("a =")
        self._a     = _dspin(*_BOUNDS_RANGE, 0.0)
        self._b_lbl = _lbl("b =")
        self._b     = _dspin(*_BOUNDS_RANGE, 1.0)
        for w in (self._a_lbl, self._a, self._b_lbl, self._b):
            brow.addWidget(w)
            w.setEnabled(False)
        brow.addStretch()
        ol.addLayout(brow)
        irow = QHBoxLayout()
        irow.addWidget(_lbl("Fit mode:"))
        self._interp = QComboBox()
        self._interp.addItems(INTERP_NAMES)
        self._interp.setCurrentText(_DEFAULT_INTERP)
        self._interp.currentTextChanged.connect(
            lambda t: self.interp_changed.emit(get_fit_mode_key(t))
        )
        irow.addWidget(self._interp)
        irow.addSpacing(12)
        irow.addWidget(_lbl("Fit n:"))
        self._n = QSpinBox()
        self._n.setRange(*_SPIN_FIT_RANGE)
        self._n.setValue(_DEFAULT_N_FIT)
        self._n.setSingleStep(100)
        self._n.setToolTip(
            "Number of points sampled from the chart fit curve for integration"
        )
        irow.addWidget(self._n)
        irow.addSpacing(12)
        irow.addWidget(_lbl("Method:"))
        self._method = QComboBox()
        self._method.addItems(METHOD_NAMES)
        irow.addWidget(self._method)
        irow.addStretch()
        ol.addLayout(irow)
        lay.addWidget(og)

    def _toggle_bounds(self, state: int) -> None:
        en = state == Qt.Checked
        for w in (self._a_lbl, self._a, self._b_lbl, self._b):
            w.setEnabled(en)

    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        if self._use_bounds.isChecked():
            return self._a.value(), self._b.value()
        return None, None

    def get_cutoff_x(self) -> Optional[float]:
        return self._b.value() if self._use_bounds.isChecked() else None

    def get_fit_n(self) -> int:
        return self._n.value()

    def get_fit_mode_key(self) -> str:
        return get_fit_mode_key(self._interp.currentText())

    def get_method(self) -> str:
        return self._method.currentText()

    def get_raw_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        pts = self._table.get_points()
        if len(pts) < 2:
            raise ValueError("Need at least 2 data points.")
        return (
            np.array([p[0] for p in pts], dtype=np.float64),
            np.array([p[1] for p in pts], dtype=np.float64),
        )

    def to_state(self) -> dict:
        return {
            "points":     self._table.to_state(),
            "use_bounds": self._use_bounds.isChecked(),
            "a":          self._a.value(),
            "b":          self._b.value(),
            "interp":     self._interp.currentText(),
            "n_fit":      self._n.value(),
            "method":     self._method.currentText(),
        }

    def apply_state(self, s: dict) -> None:
        self._table.apply_state(s.get("points", []))
        self._use_bounds.setChecked(s.get("use_bounds", False))
        self._a.setValue(s.get("a", 0.0))
        self._b.setValue(s.get("b", 1.0))
        idx = self._interp.findText(s.get("interp", _DEFAULT_INTERP))
        if idx >= 0:
            self._interp.setCurrentIndex(idx)
        self._n.setValue(s.get("n_fit", s.get("n_resample", _DEFAULT_N_FIT)))
        idx2 = self._method.findText(s.get("method", _DEFAULT_METHOD))
        if idx2 >= 0:
            self._method.setCurrentIndex(idx2)


class AreaCalculatorPanel(QWidget):
    """
    Main dock panel for the Area Calculator plugin.

    Expression tab: evaluates f(x) analytically then integrates.
    Data tab: pushes raw (x,y) into the ChartOverlay; a _FitItem on the chart
    performs interpolation natively, then getData() retrieves the dense curve
    for numerical integration — no scipy/numpy interpolation in-plugin.
    """

    def __init__(self, context: "AppContext", parent: QWidget = None) -> None:
        super().__init__(parent)
        self._ctx     = context
        self._overlay = ChartOverlay(context.chart)
        self._last_area: Optional[float] = None
        self.setMinimumWidth(340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        hdr = QWidget()
        hdr.setStyleSheet("background:#1e3a5f;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 6, 10, 6)
        title = QLabel("Area Calculator")
        hl.addWidget(title)
        hl.addStretch()
        self._show_chart = QCheckBox("Show on chart")
        self._show_chart.setChecked(True)
        hl.addWidget(self._show_chart)
        lay.addWidget(hdr)
        self._tabs = QTabWidget()
        self._expr_tab = _ExprTab()
        self._data_tab = _DataTab()
        self._tabs.addTab(self._expr_tab, "Expression")
        self._tabs.addTab(self._data_tab, "Data Table")
        lay.addWidget(self._tabs, 1)
        brow = QHBoxLayout()
        brow.setContentsMargins(8, 6, 8, 6)
        brow.setSpacing(8)
        self._calc_btn = _btn("Calculate Area", "#16a34a", "#15803d")
        self._calc_btn.setFixedHeight(32)
        self._calc_btn.clicked.connect(self._compute)
        self._copy_btn = _btn("Copy", "#6b7280", "#4b5563")
        self._copy_btn.setFixedHeight(32)
        self._copy_btn.clicked.connect(self._copy)
        self._clear_btn = _btn("Clear Chart", "#6b7280", "#4b5563")
        self._clear_btn.setFixedHeight(32)
        self._clear_btn.clicked.connect(self._clear_overlay)
        brow.addWidget(self._calc_btn, 1)
        brow.addWidget(self._copy_btn)
        brow.addWidget(self._clear_btn)
        lay.addLayout(brow)
        self._result = _ResultBox()
        lay.addWidget(self._result)
        self._expr_tab.compute_requested.connect(self._compute)
        self._data_tab.compute_requested.connect(self._compute)
        self._data_tab.interp_changed.connect(self._on_interp_changed)

    def _on_interp_changed(self, mode_key: str) -> None:
        self._overlay.set_fit_mode(mode_key)

    def _compute(self) -> None:
        self._result.clear()
        try:
            if self._tabs.currentIndex() == 0:
                self._compute_expr()
            else:
                self._compute_data()
        except Exception as exc:
            self._last_area = None
            self._result.show_error(str(exc))

    def _compute_expr(self) -> None:
        area, a, b, n, method, xs, ys = self._expr_tab.compute()
        self._last_area = area
        self._result.show_result(area, a, b, n, method)
        if self._show_chart.isChecked():
            self._overlay.update_raw_data(xs, ys)
            self._overlay.set_fit_mode("linear")
            self._overlay.set_fit_label(f"f(x) [{a:.3g}, {b:.3g}]")
            self._overlay.hide_scatter()
            self._overlay.update_cutline(None)
            self._ctx.chart.update()
        self._show_status(area, a, b, method)

    def _compute_data(self) -> None:
        raw_xs, raw_ys = self._data_tab.get_raw_arrays()
        a, b           = self._data_tab.get_bounds()
        method         = self._data_tab.get_method()
        n_fit          = self._data_tab.get_fit_n()
        interp_name    = self._data_tab._interp.currentText()
        mode_key       = self._data_tab.get_fit_mode_key()
        self._overlay.set_fit_mode(mode_key)
        self._overlay.update_raw_data(raw_xs, raw_ys)
        self._overlay.update_scatter(raw_xs, raw_ys)
        self._overlay.update_cutline(b)
        fit_xs, fit_ys = self._overlay.get_fit_data(
            x_lo=a, x_hi=b, n_pts=n_fit,
        )
        if not fit_xs:
            raise ValueError(
                "Fit returned no data. Check that data range covers [a, b]."
            )
        area = integrate_from_fit_data(fit_xs, fit_ys, method)
        lo   = fit_xs[0]
        hi   = fit_xs[-1]
        self._last_area = area
        self._overlay.set_fit_label(f"Fit [{lo:.3g}, {hi:.3g}]")
        self._result.show_result(area, lo, hi, n_fit, method, interp_name)
        if self._show_chart.isChecked():
            self._ctx.chart.update()
        self._show_status(area, lo, hi, method)

    def _show_status(self, area: float, lo: float, hi: float, method: str) -> None:
        self._ctx.show_status(
            f"Area = {area:.6g}  [{lo:.4g}, {hi:.4g}]  {method}", 4000
        )

    def _copy(self) -> None:
        if self._last_area is not None:
            QApplication.clipboard().setText(str(self._last_area))

    def _clear_overlay(self) -> None:
        self._overlay.hide_all()
        self._ctx.chart.update()

    def cleanup(self) -> None:
        self._overlay.cleanup()
        self._ctx.chart.update()

    def to_state(self) -> dict:
        return {
            "tab":  self._tabs.currentIndex(),
            "expr": self._expr_tab.to_state(),
            "data": self._data_tab.to_state(),
        }

    def apply_state(self, s: dict) -> None:
        self._tabs.setCurrentIndex(s.get("tab", 0))
        if "expr" in s:
            self._expr_tab.apply_state(s["expr"])
        if "data" in s:
            self._data_tab.apply_state(s["data"])