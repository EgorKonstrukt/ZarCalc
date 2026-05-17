from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from pyqt5_chart_widget import ChartWidget
    from pyqt5_chart_widget.items import _LineItem, _ScatterItem, _InfLine, _FitItem

_RAW_COLOR    = "#e67e22"
_FIT_COLOR    = "#27ae60"
_CUT_COLOR    = "#e74c3c"
_RAW_WIDTH    = 1
_FIT_WIDTH    = 2
_SCATTER_SZ   = 8
_CURVE_ALPHA  = 180
_FIT_N_PTS    = 800


class ChartOverlay:
    """
    Manages chart items injected by the Area Calculator plugin.

    Strategy: raw scatter + raw thin line are set via setData; a _FitItem
    attached to that raw line provides native chart-level interpolation
    (pchip, spline, poly*, linear).  Integration uses FitItem.getData()
    to retrieve the dense evaluated curve without re-implementing any math.
    """

    def __init__(self, chart: "ChartWidget") -> None:
        self._chart = chart
        self._raw_line:  Optional[_LineItem]    = None
        self._fit_item:  Optional[_FitItem]     = None
        self._scatter:   Optional[_ScatterItem] = None
        self._cutline:   Optional[_InfLine]     = None
        self._fit_mode:  str                    = "pchip"
        self._active:    bool                   = False

    def _ensure_items(self) -> None:
        if self._active:
            return
        self._raw_line = self._chart.plot(
            color=_RAW_COLOR, width=_RAW_WIDTH, label="Data (raw)"
        )
        self._raw_line.setRawVisible(False)
        self._fit_item = self._chart.addFit(
            self._raw_line,
            mode_key=self._fit_mode,
            color=_FIT_COLOR,
            width=_FIT_WIDTH,
            label="Area curve",
        )
        self._scatter = self._chart.addScatter(
            size=_SCATTER_SZ, color=_RAW_COLOR, label="Data pts"
        )
        self._cutline = self._chart.addLine(
            x=0.0, color=_CUT_COLOR, width=2, dashed=True
        )
        self._cutline.setVisible(False)
        self._active = True

    def set_fit_mode(self, mode_key: str) -> None:
        """Switch the interpolation algorithm on the live _FitItem."""
        self._fit_mode = mode_key
        if self._fit_item is not None:
            self._fit_item.setModeKey(mode_key)

    def update_raw_data(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Push raw (x,y) pairs into the source line; the fit redraws automatically."""
        self._ensure_items()
        fin = np.isfinite(xs) & np.isfinite(ys)
        self._raw_line.setData(
            xs=xs[fin].tolist(),
            ys=ys[fin].tolist(),
        )

    def update_scatter(self, xs: np.ndarray, ys: np.ndarray) -> None:
        self._ensure_items()
        fin = np.isfinite(xs) & np.isfinite(ys)
        if fin.any():
            self._scatter.setData(xs=xs[fin].tolist(), ys=ys[fin].tolist())
            self._scatter.setVisible(True)
        else:
            self._scatter.setData(xs=[], ys=[])
            self._scatter.setVisible(False)

    def hide_scatter(self) -> None:
        if self._scatter is not None:
            self._scatter.setData(xs=[], ys=[])
            self._scatter.setVisible(False)

    def update_cutline(self, x: Optional[float]) -> None:
        self._ensure_items()
        if x is not None:
            self._cutline.setValue(x)
            self._cutline.setVisible(True)
        else:
            self._cutline.setVisible(False)

    def set_fit_label(self, label: str) -> None:
        if self._fit_item is not None:
            self._fit_item.setLabel(label)

    def get_fit_data(
        self,
        x_lo: Optional[float] = None,
        x_hi: Optional[float] = None,
        n_pts: int = _FIT_N_PTS,
    ) -> Tuple[List[float], List[float]]:
        """
        Retrieve dense fit curve data for numerical integration.
        Returns (xs, ys) as plain Python lists.
        """
        if self._fit_item is None:
            return [], []
        kw: dict = {"n_pts": n_pts}
        if x_lo is not None:
            kw["x_lo"] = x_lo
        if x_hi is not None:
            kw["x_hi"] = x_hi
        return self._fit_item.getData(**kw)

    def evaluate_fit(self, x: float) -> Optional[float]:
        """Evaluate the fit at a single x value."""
        if self._fit_item is None:
            return None
        return self._fit_item.evaluate(x)

    def hide_all(self) -> None:
        for item in (self._raw_line, self._fit_item, self._scatter, self._cutline):
            if item is not None:
                item.setVisible(False)

    def cleanup(self) -> None:
        for item in (self._cutline, self._scatter, self._fit_item, self._raw_line):
            if item is not None:
                try:
                    self._chart.removeItem(item)
                except Exception:
                    pass
        self._raw_line = self._fit_item = self._scatter = self._cutline = None
        self._active = False