from __future__ import annotations
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt

if TYPE_CHECKING:
    from core import AppContext

_DEFAULT_COLOR = "#3498db"
_DEFAULT_WIDTH = 2
_MIN_INTERVAL_MS = 8


def _force_repaint_widget(w):
    if w is None:
        return
    try:
        scene = None
        if hasattr(w, "scene") and callable(w.scene):
            scene = w.scene()
        if scene is not None:
            scene.update()
        vp = None
        if hasattr(w, "viewport") and callable(w.viewport):
            vp = w.viewport()
        if vp is not None:
            vp.update()
            vp.repaint()
        else:
            w.update()
            w.repaint()
    except Exception:
        pass


class PlotWindow(QDialog):
    def __init__(self, title: str = "Script Plot", parent=None):
        super().__init__(parent)
        from pyqt5_chart_widget import ChartWidget
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._chart = ChartWidget(
            show_toolbar=True, show_legend=True,
            show_sidebar=False, threaded_fit=False, anim_duration=0,
        )
        lay.addWidget(self._chart)
        self._lines: Dict[str, Any] = {}

    def _repaint(self):
        _force_repaint_widget(self._chart)

    @property
    def chart(self):
        return self._chart

    def plot(self, xs: List[float], ys: List[float], label: str = "",
             color: str = _DEFAULT_COLOR, width: int = _DEFAULT_WIDTH) -> Any:
        from PyQt5.QtGui import QColor
        key = label or f"_l{len(self._lines)}"
        if key not in self._lines:
            self._lines[key] = self._chart.plot(label=label, color=color, width=width)
        line = self._lines[key]
        line.pen.setColor(QColor(color))
        line.pen.setWidth(width)
        if label:
            line.setLabel(label)
        line.setData(xs=list(xs), ys=list(ys))
        line.setVisible(True)
        self._repaint()
        return line

    def scatter(self, xs: List[float], ys: List[float], label: str = "",
                color: str = "#e74c3c", size: int = 6) -> Any:
        from PyQt5.QtGui import QColor
        key = f"_sc_{label or len(self._lines)}"
        if key not in self._lines:
            self._lines[key] = self._chart.plot(label=label, color=color, width=1)
        line = self._lines[key]
        line.pen.setColor(QColor(color))
        line.setData(xs=list(xs), ys=list(ys))
        line.setVisible(True)
        self._repaint()
        return line

    def clear(self):
        for line in self._lines.values():
            try:
                line.setData(xs=[], ys=[])
            except Exception:
                pass
        self._lines.clear()
        self._repaint()

    def set_title(self, title: str):
        self.setWindowTitle(title)

    def set_axis_labels(self, x_label: str = "", y_label: str = ""):
        if x_label:
            self._chart.setLabel("bottom", x_label)
        if y_label:
            self._chart.setLabel("left", y_label)

    def autofit(self):
        self._chart.autofit()


class AnimHandle:
    def __init__(self, api: "ScriptAPI", handle_id: int):
        self._api = api
        self._id = handle_id

    def stop(self):
        self._api._stop_anim(self._id)

    @property
    def t(self) -> float:
        return self._api._get_anim_t(self._id)

    @property
    def running(self) -> bool:
        return self._id in self._api._anim_records


class ScriptAPI:
    def __init__(self, context: "AppContext", owner_row):
        self._ctx = context
        self._row = owner_row
        self._anim_records: Dict[int, Dict] = {}
        self._anim_timers: Dict[int, QTimer] = {}
        self._anim_counter = 0
        self._plot_windows: List[PlotWindow] = []
        self._data_store: Dict[str, Any] = {}
        self._event_hooks: Dict[str, List[Callable]] = {}

    def _console_api(self):
        return self._ctx.get_service("console_api")

    def _tab_id(self) -> str:
        return self._row._tab_id or "__repl__"

    def _get_chart(self):
        return self._ctx.chart

    def _force_chart_update(self):
        _force_repaint_widget(self._get_chart())

    def _force_all_windows_update(self):
        for win in self._plot_windows:
            try:
                win._repaint()
            except Exception:
                pass

    def plot(self, xs, ys, label: str = "", color: str = _DEFAULT_COLOR,
             width: int = _DEFAULT_WIDTH) -> Any:
        from PyQt5.QtGui import QColor
        key = f"_sr_{id(self._row)}_{label or id((xs, ys))}"
        lines = self._row._script_lines
        if key not in lines:
            lines[key] = self._get_chart().plot(label=label or "script", color=color, width=width)
        line = lines[key]
        line.pen.setColor(QColor(color))
        line.pen.setWidth(width)
        if label:
            line.setLabel(label)
        line.setData(xs=list(xs), ys=list(ys))
        line.setVisible(True)
        self._force_chart_update()
        return line

    def plot_parametric(self, xs, ys, label: str = "", color: str = "#9b59b6",
                        width: int = _DEFAULT_WIDTH) -> Any:
        return self.plot(xs, ys, label=label, color=color, width=width)

    def plot_polar(self, thetas: List[float], rs: List[float], label: str = "",
                   color: str = "#1abc9c", width: int = _DEFAULT_WIDTH) -> Any:
        xs = [r * math.cos(t) for t, r in zip(thetas, rs)]
        ys = [r * math.sin(t) for t, r in zip(thetas, rs)]
        return self.plot(xs, ys, label=label, color=color, width=width)

    def vline(self, x: float, label: str = "", color: str = "#e74c3c", width: int = 1) -> Any:
        key = f"_vl_{id(self._row)}_{label or x}"
        lines = self._row._script_lines
        if key not in lines:
            lines[key] = self._get_chart().plot(label=label, color=color, width=width)
        chart = self._get_chart()
        vp = chart.getViewBox() if hasattr(chart, "getViewBox") else None
        y_range = vp.viewRange()[1] if vp else [-1e6, 1e6]
        lines[key].setData(xs=[x, x], ys=y_range)
        lines[key].setVisible(True)
        self._force_chart_update()
        return lines[key]

    def hline(self, y: float, label: str = "", color: str = "#e74c3c", width: int = 1) -> Any:
        key = f"_hl_{id(self._row)}_{label or y}"
        lines = self._row._script_lines
        if key not in lines:
            lines[key] = self._get_chart().plot(label=label, color=color, width=width)
        chart = self._get_chart()
        vp = chart.getViewBox() if hasattr(chart, "getViewBox") else None
        x_range = vp.viewRange()[0] if vp else [-1e6, 1e6]
        lines[key].setData(xs=x_range, ys=[y, y])
        lines[key].setVisible(True)
        self._force_chart_update()
        return lines[key]

    def clear_plots(self):
        for line in list(self._row._script_lines.values()):
            try:
                self._get_chart().removeItem(line)
            except Exception:
                pass
        self._row._script_lines.clear()
        self._force_chart_update()

    def add_function(self, expr: str, mode: str = "y=f(x)",
                     color: str = "#e74c3c", width: int = _DEFAULT_WIDTH):
        from constants import COLORS
        panel = self._ctx.panel
        idx = len(panel.func_rows)
        state = {
            "expr": expr, "mode": mode, "expr2": "",
            "color": color or COLORS[idx % len(COLORS)],
            "width": width, "enabled": True, "type": "function",
        }
        return panel.add_function_from_state(state)

    def add_param(self, name: str, lo: float = -5.0, hi: float = 5.0, val: float = 1.0,
                  speed: float = 1.0):
        panel = self._ctx.panel
        if name not in panel._param_widgets:
            panel.add_param(name, record=False, state={
                "name": name, "lo": lo, "hi": hi, "val": val,
                "speed": speed, "anim_mode": "loop", "type": "param",
            })

    def set_param(self, name: str, val: float):
        panel = self._ctx.panel
        pw = panel._param_widgets.get(name)
        if pw is not None and hasattr(pw, "set_value"):
            pw.set_value(val)

    def get_param(self, name: str, default: float = 0.0) -> float:
        return self._ctx.panel.get_params().get(name, default)

    def get_all_params(self) -> Dict[str, float]:
        return dict(self._ctx.panel.get_params())

    def get_t(self) -> float:
        anim = getattr(self._ctx.panel, "anim_panel", None)
        return anim.get_t() if anim is not None else 0.0

    def get_viewport(self) -> Tuple[float, float, float, float]:
        chart = self._get_chart()
        vp = chart.getViewBox() if hasattr(chart, "getViewBox") else None
        if vp is None:
            return (-10.0, 10.0, -10.0, 10.0)
        xr, yr = vp.viewRange()
        return (xr[0], xr[1], yr[0], yr[1])

    def set_viewport(self, x0: float, x1: float, y0: float, y1: float):
        chart = self._get_chart()
        if hasattr(chart, "setRange"):
            chart.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def new_window(self, title: str = "Script Plot") -> PlotWindow:
        win = PlotWindow(title, self._ctx.main_window)
        self._plot_windows.append(win)
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def animate(self, callback: Callable[[float], None],
                fps: int = 30, duration_ms: int = 0) -> AnimHandle:
        handle_id = self._anim_counter
        self._anim_counter += 1
        interval = max(_MIN_INTERVAL_MS, 1000 // max(1, fps))
        record: Dict = {"elapsed_ms": 0}
        self._anim_records[handle_id] = record
        timer = QTimer(self._row)
        timer.setInterval(interval)

        def _tick():
            record["elapsed_ms"] += interval
            if duration_ms > 0 and record["elapsed_ms"] >= duration_ms:
                timer.stop()
                self._anim_records.pop(handle_id, None)
                self._anim_timers.pop(handle_id, None)
                self._fire_event("anim_done", handle_id)
                return
            try:
                callback(record["elapsed_ms"] / 1000.0)
            except Exception as exc:
                import traceback as _tb
                self._row._set_status(f"Anim error: {exc}", error=True)
                self._row._log_error(_tb.format_exc())
                timer.stop()
                self._anim_records.pop(handle_id, None)
                self._anim_timers.pop(handle_id, None)
                return
            self._force_chart_update()
            self._force_all_windows_update()

        timer.timeout.connect(_tick)
        self._anim_timers[handle_id] = timer
        timer.start()
        return AnimHandle(self, handle_id)

    def stop_all_anims(self):
        for handle_id in list(self._anim_timers.keys()):
            self._stop_anim(handle_id)

    def schedule_once(self, callback: Callable, delay_ms: int = 0):
        timer = QTimer(self._row)
        timer.setSingleShot(True)
        timer.setInterval(max(0, delay_ms))
        def _run():
            try:
                callback()
            except Exception:
                import traceback as _tb
                self._row._log_error(_tb.format_exc())
            self._force_chart_update()
        timer.timeout.connect(_run)
        timer.start()

    def linspace(self, start: float, stop: float, n: int) -> List[float]:
        from math_engine import linspace as _ls
        return _ls(start, stop, n)

    def arange(self, start: float, stop: float, step: float = 1.0) -> List[float]:
        result = []
        x = start
        if step > 0:
            while x < stop:
                result.append(x)
                x += step
        elif step < 0:
            while x > stop:
                result.append(x)
                x += step
        return result

    def zeros(self, n: int) -> List[float]:
        return [0.0] * n

    def ones(self, n: int) -> List[float]:
        return [1.0] * n

    def map_fn(self, fn: Callable, xs: List[float]) -> List[float]:
        return [fn(x) for x in xs]

    def zip_xy(self, xs: List[float], ys: List[float]) -> List[Tuple[float, float]]:
        return list(zip(xs, ys))

    def status(self, msg: str, timeout_ms: int = 4000):
        self._ctx.show_status(msg, timeout_ms)

    def log(self, *args):
        msg = " ".join(str(a) for a in args)
        api = self._console_api()
        if api is not None:
            api.write(msg, tab_id=self._tab_id(), source=self._row._script_name())

    def log_warn(self, *args):
        msg = " ".join(str(a) for a in args)
        api = self._console_api()
        if api is not None:
            api.log_warn(msg, tab_id=self._tab_id(), source=self._row._script_name())

    def log_error(self, *args):
        msg = " ".join(str(a) for a in args)
        api = self._console_api()
        if api is not None:
            api.write_stderr(msg, tab_id=self._tab_id(), source=self._row._script_name())

    def replot(self):
        self._ctx.request_replot()

    def autofit(self):
        chart = self._get_chart()
        if hasattr(chart, "autofit"):
            chart.autofit()

    def get_time(self) -> float:
        return time.perf_counter()

    def store(self, key: str, value: Any):
        self._data_store[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        return self._data_store.get(key, default)

    def store_keys(self) -> List[str]:
        return list(self._data_store.keys())

    def on(self, event: str, callback: Callable):
        self._event_hooks.setdefault(event, []).append(callback)

    def _fire_event(self, event: str, *args):
        for cb in self._event_hooks.get(event, []):
            try:
                cb(*args)
            except Exception:
                pass

    def _stop_anim(self, handle_id: int):
        timer = self._anim_timers.pop(handle_id, None)
        if timer:
            timer.stop()
        self._anim_records.pop(handle_id, None)

    def _get_anim_t(self, handle_id: int) -> float:
        rec = self._anim_records.get(handle_id)
        return rec["elapsed_ms"] / 1000.0 if rec else 0.0

    def cleanup(self):
        for timer in list(self._anim_timers.values()):
            try:
                timer.stop()
            except Exception:
                pass
        self._anim_timers.clear()
        self._anim_records.clear()
        self.clear_plots()
        for win in list(self._plot_windows):
            try:
                win.close()
            except Exception:
                pass
        self._plot_windows.clear()
        self._data_store.clear()
        self._event_hooks.clear()