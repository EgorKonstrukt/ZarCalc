from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction, QApplication, QHBoxLayout, QLabel,
    QMainWindow, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from pyqt5_chart_widget import ChartWidget
from constants import APP_NAME, APP_VERSION
from core.panels import FunctionPanel
from plotter import Plotter
from history import History
from io_manager import IoManager
from config import Config
from settings_dialog import SettingsDialog
from core.plugins.app_context import AppContext
from core.plugins.plugin_manager import PluginManager
from core.plugins.plugin_manager_dialog import PluginManagerDialog
from compute_pool import get_pool
import math_engine

_BASE_DIR    = Path(__file__).parent
_PLUGINS_DIR = _BASE_DIR / "plugins"

_INITIAL_FUNCTION = {
    "expr": "sin(x)", "mode": "y=f(x)", "expr2": "",
    "color": "#3498db", "width": 2, "enabled": True, "type": "function",
}
_EMPTY_FUNCTION = {
    "expr": "", "mode": "y=f(x)", "expr2": "",
    "color": "#3498db", "width": 2, "enabled": True, "type": "function",
}

from core.plugins.settings_registry import SettingsRegistry


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._cfg = Config()
        math_engine.set_use_numpy(self._cfg.use_numpy)
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1280, 820)
        self._history = History()
        self._fps_times: list = []
        self._pool = get_pool()
        self._build_ui()
        self._plotter = Plotter(self._chart, self._panel)
        self._io = IoManager(self._panel, self)
        self._upd_timer = QTimer(self)
        self._upd_timer.setSingleShot(True)
        self._upd_timer.timeout.connect(self._replot)
        self._panel.update_requested.connect(self._schedule)
        self._panel.anim_update_requested.connect(self._replot)
        self._context = AppContext(
            main_window=self,
            chart=self._chart,
            panel=self._panel,
            history=self._history,
            config=self._cfg,
        )
        self._panel.set_context(self._context)
        self._plugin_manager = PluginManager(
            plugins_dir=_PLUGINS_DIR,
            state_dir=_BASE_DIR,
        )

        self._build_menu()
        self._context.register_menu("File",    self._file_menu)
        self._context.register_menu("Edit",    self._edit_menu)
        self._context.register_menu("View",    self._view_menu)
        self._context.register_menu("Plugins", self._plugin_menu)
        self._panel.settings.infinite_changed.connect(self._on_infinite_changed)
        self._chart.onViewportChanged(self._on_viewport_changed)
        self._apply_config()
        self._panel.add_function_from_state(_INITIAL_FUNCTION)
        self._settings_registry = SettingsRegistry()
        self._context.register_service("settings_registry", self._settings_registry)

        self._plugin_manager.initialise(self._context)
        self._integrate_plugins()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._context, self)
        dlg.settings_applied.connect(self._apply_config)
        dlg.exec_()

    def closeEvent(self, event) -> None:
        self._upd_timer.stop()
        self._plotter._drain_timer.stop()
        self._plugin_manager.shutdown(self._context)
        console = self._context.get_service("script_console")
        if console is not None:
            console.close()
        self._pool.shutdown()
        super().closeEvent(event)

    def _integrate_plugins(self) -> None:
        for p in self._plugin_manager.panel_plugins():
            self._panel.register_plugin_item_type(p.menu_label, p)
        for sp in self._plugin_manager.sidebar_plugins():
            widget = sp.create_panel(self._context)
            if widget is not None:
                self._add_sidebar_widget(widget)
        self._wire_console_menu()

    def _wire_console_menu(self) -> None:
        console = self._context.get_service("script_console")
        if console is None:
            return
        act = QAction("Script &Console", self)
        act.setShortcut(QKeySequence("Ctrl+`"))
        act.setShortcutContext(Qt.ApplicationShortcut)
        act.setCheckable(True)
        act.setChecked(False)
        act.triggered.connect(lambda checked: console.setVisible(checked))
        console.visibilityChanged.connect(act.setChecked)
        self.addAction(act)
        self._view_menu.addAction(act)

    def _add_sidebar_widget(self, widget: QWidget) -> None:
        self._bottom_area.layout().addWidget(widget)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self._panel = FunctionPanel(None, self._history)
        self._chart = ChartWidget(
            show_toolbar=True, show_legend=True,
            show_sidebar=False, threaded_fit=False, anim_duration=60,
        )
        self._panel.bind_chart(self._chart)
        self._panel.settings.changed.connect(self._schedule)
        self._panel.deriv_panel.changed.connect(self._schedule)
        self._chart.setLabel("left", "y")
        self._chart.setLabel("bottom", "x")
        self._ruler = self._chart.addRuler(color="#e74c3c", width=2)
        self._ruler.changed = self._on_ruler_changed
        self._info_bar = QLabel()
        self._info_bar.setAlignment(Qt.AlignCenter)
        self._ruler_label = QLabel()
        self._ruler_label.setAlignment(Qt.AlignRight)
        self._ruler_label.setFixedWidth(320)
        self._ruler_label.setVisible(False)
        self._fps_label = QLabel("-- fps")
        self._fps_label.setAlignment(Qt.AlignRight)
        self._fps_label.setFixedWidth(70)
        self._vp_label = QLabel()
        self._vp_label.setAlignment(Qt.AlignRight)
        self._vp_label.setFixedWidth(340)
        right = QWidget()
        rl = QHBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._chart)
        self._left_wrap = QWidget()
        self._left_wrap.setFixedWidth(self._cfg.panel_width)
        ll = QVBoxLayout(self._left_wrap)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        ll.addWidget(self._panel)
        self._bottom_area = QWidget()
        self._bottom_area.setObjectName("sidebar_area")
        ba_lay = QVBoxLayout(self._bottom_area)
        ba_lay.setContentsMargins(0, 0, 0, 0)
        ba_lay.setSpacing(0)
        splitter.addWidget(self._left_wrap)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        central = QWidget()
        cl = QHBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(splitter)
        self.setCentralWidget(central)
        sb = QStatusBar()
        sb.addPermanentWidget(self._fps_label)
        sb.addPermanentWidget(self._ruler_label)
        sb.addPermanentWidget(self._vp_label)
        sb.addPermanentWidget(self._info_bar)
        sb.showMessage(
            "+ Add to add items  |  + param to add slider  |  Ctrl+Z/Y = undo/redo"
        )
        self.setStatusBar(sb)

    def _build_menu(self) -> None:
        mb = self.menuBar()
        def act(label, shortcut, fn):
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
                a.setShortcutContext(Qt.ApplicationShortcut)
            a.triggered.connect(fn)
            self.addAction(a)
            return a
        self._file_menu = mb.addMenu("&File")
        self._file_menu.addAction(act("&New",        "Ctrl+N",       self._new_session))
        self._file_menu.addAction(act("&Open...",    "Ctrl+O",       self._io.load))
        self._file_menu.addAction(act("&Save",       "Ctrl+S",       self._io.save))
        self._file_menu.addAction(act("Save &As...", "Ctrl+Shift+S", self._io.save_as))
        self._file_menu.addSeparator()
        self._file_menu.addAction(act("&Quit",       "Ctrl+Q",       QApplication.quit))
        self._edit_menu = mb.addMenu("&Edit")
        self._undo_act = act("&Undo", "Ctrl+Z", self._undo)
        self._redo_act = act("&Redo", "Ctrl+Y", self._redo)
        self._edit_menu.addAction(self._undo_act)
        self._edit_menu.addAction(self._redo_act)
        self._view_menu = mb.addMenu("&View")
        self._view_menu.addAction(act("&Autofit",   "Ctrl+F", self._chart.autofit))
        self._view_menu.addAction(act("&Clear All", None,     self._panel._clear_all))
        self._view_menu.addSeparator()
        self._ruler_action = QAction("&Ruler", self)
        self._ruler_action.setShortcut(QKeySequence("Ctrl+R"))
        self._ruler_action.setShortcutContext(Qt.ApplicationShortcut)
        self._ruler_action.setCheckable(True)
        self._ruler_action.setChecked(False)
        self._ruler_action.toggled.connect(self._toggle_ruler)
        self.addAction(self._ruler_action)
        self._view_menu.addAction(self._ruler_action)
        sm = mb.addMenu("&Settings")
        sm.addAction(act("&Preferences...", "Ctrl+,", self._open_settings))
        self._plugin_menu = mb.addMenu("&Plugins")
        self._plugin_menu.addAction(
            act("&Manage Plugins...", None, self._open_plugin_manager)
        )
        self._plugin_menu.addSeparator()

    def _open_plugin_manager(self) -> None:
        dlg = PluginManagerDialog(self._plugin_manager, self)
        dlg.exec_()
        self._integrate_plugins()

    def _apply_config(self) -> None:
        math_engine.set_use_numpy(self._cfg.use_numpy)
        self._panel.set_anim_interval(self._cfg.anim_interval_ms)
        self._left_wrap.setFixedWidth(self._cfg.panel_width)
        self._fps_label.setVisible(self._cfg.show_fps)

    def _undo(self) -> None:
        self._history.undo()
        self._schedule()

    def _redo(self) -> None:
        self._history.redo()
        self._schedule()

    def _on_infinite_changed(self, enabled: bool) -> None:
        self._chart.setAutofitEnabled(not enabled)
        if not enabled:
            self._chart.autofit()
        self._schedule()

    def _on_viewport_changed(self, x0, x1, y0, y1) -> None:
        span_x = x1 - x0
        span_y = y1 - y0
        self._vp_label.setText(
            f"x=[{x0:.3g}, {x1:.3g}]  y=[{y0:.3g}, {y1:.3g}]  "
            f"Δx={span_x:.3g}  Δy={span_y:.3g}"
        )

    def _toggle_ruler(self, checked: bool) -> None:
        self._ruler.setVisible(checked)
        self._ruler_label.setVisible(checked)
        if not checked:
            self._ruler_label.clear()

    def _on_ruler_changed(self) -> None:
        r = self._ruler
        self._ruler_label.setText(
            f"Ruler  d={r.distance:.4g}  dx={r.dx:.4g}  dy={r.dy:.4g}  angle={r.angle_deg:.1f}deg"
        )

    def _schedule(self) -> None:
        self._upd_timer.start(self._cfg.replot_delay_ms)

    def _replot(self) -> None:
        result = self._plotter.replot()
        if result:
            total, x_min, x_max, n_fn = result
            self._info_bar.setText(
                f"~{total:,} pts  x=[{x_min:.2f}, {x_max:.2f}]  {n_fn} fn"
            )
        if self._cfg.show_fps:
            now = time.perf_counter()
            self._fps_times.append(now)
            cutoff = now - 1.0
            self._fps_times = [t for t in self._fps_times if t > cutoff]
            self._fps_label.setText(f"{len(self._fps_times)} fps")

    def _new_session(self) -> None:
        self._plotter.sync_fn_items()
        self._chart.clearAll()
        self._panel._clear_all()
        self._panel.add_function_from_state(_EMPTY_FUNCTION)