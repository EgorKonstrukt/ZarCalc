from __future__ import annotations
from core.plugins.plugin_base import PluginMeta, DockPlugin
from core.plugins.app_context import AppContext
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget

PLUGIN_META = PluginMeta(
    id="area_calculator",
    name="Area Calculator",
    version="1.0.0",
    author="ZarCalc",
    description="Numerical area integration with live chart overlay.",
)


class _AreaCalculatorPlugin(DockPlugin):
    meta = PLUGIN_META
    DOCK_ID = "area_calculator"
    DEFAULT_AREA = Qt.RightDockWidgetArea
    DEFAULT_VISIBLE = False

    def __init__(self) -> None:
        self._dock: QDockWidget | None = None
        self._panel = None
        self._action = None

    def create_dock(self, context: "AppContext") -> QDockWidget:
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QAction
        from .panel import AreaCalculatorPanel
        self._panel = AreaCalculatorPanel(context)
        dock = QDockWidget("Area Calculator", context.main_window)
        dock.setObjectName("area_calculator_dock")
        dock.setWidget(self._panel)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self._dock = dock
        act = QAction("&Area Calculator", context.main_window)
        act.setShortcut(QKeySequence("Ctrl+Shift+A"))
        act.setShortcutContext(Qt.ApplicationShortcut)
        act.setCheckable(True)
        act.setChecked(False)
        act.toggled.connect(self._dock.setVisible)
        self._dock.visibilityChanged.connect(act.setChecked)
        context.main_window.addAction(act)
        self._action = act
        view_menu = context.get_menu("View")
        if view_menu is not None:
            view_menu.addSeparator()
            view_menu.addAction(act)
        context.register_service("area_panel", self._panel)
        return dock

    def on_load(self, context: "AppContext") -> None:
        if self._panel is not None:
            context.register_service("area_panel", self._panel)

    def on_unload(self, context: "AppContext") -> None:
        if self._panel is not None:
            self._panel.cleanup()
        if self._dock is not None:
            self._dock.close()


def get_plugin() -> _AreaCalculatorPlugin:
    return _AreaCalculatorPlugin()