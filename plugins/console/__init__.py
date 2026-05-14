from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt5.QtCore import Qt
from core.plugins.plugin_base import DockPlugin, PluginMeta
if TYPE_CHECKING:
    from PyQt5.QtWidgets import QDockWidget
    from core.plugins.app_context import AppContext

PLUGIN_META = PluginMeta(
    id="zarcalc.console",
    name="Console",
    version="1.0.0",
    author="ZarCalc",
    description="Python REPL console with per-script tabs, autocomplete, and debug tools.",
    dependencies=[],
)


class ConsolePlugin(DockPlugin):
    """DockPlugin that provides the main Python console dock."""

    meta = PLUGIN_META
    DOCK_ID = "zarcalc.console"
    DEFAULT_AREA = 8
    DEFAULT_FLOATING = False
    DEFAULT_VISIBLE = True

    def __init__(self) -> None:
        self._api = None
        self._dock = None

    def create_dock(self, context: "AppContext") -> "QDockWidget":
        from .console_api import ConsoleAPI
        from .console_dock import ConsoleDock
        self._api = ConsoleAPI()
        self._api.set_app_namespace({
            "context": context,
            "chart": context.chart,
            "panel": context.panel,
            "config": context.config,
            "history": context.history,
        })
        self._dock = ConsoleDock(self._api, context.main_window)
        context.register_service("console_api", self._api)
        context.register_service("console_dock", self._dock)
        self._wire_view_menu(context)
        return self._dock

    def _wire_view_menu(self, context: "AppContext") -> None:
        from PyQt5.QtWidgets import QAction
        from PyQt5.QtGui import QKeySequence
        menu = context.get_menu("View")
        if menu is None or self._dock is None:
            return
        act = QAction("Python &Console", context.main_window)
        act.setShortcut(QKeySequence("Ctrl+`"))
        act.setCheckable(True)
        act.setChecked(self._dock.isVisible())
        act.triggered.connect(lambda checked: self._dock.setVisible(checked))
        self._dock.visibilityChanged.connect(act.setChecked)
        menu.addAction(act)

    def on_load(self, context: "AppContext") -> None:
        pass

    def on_unload(self, context: "AppContext") -> None:
        if self._api:
            self._api.shutdown()
        if self._dock:
            self._dock.close()


def get_plugin() -> ConsolePlugin:
    return ConsolePlugin()