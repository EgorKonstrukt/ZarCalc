import json
from typing import TYPE_CHECKING
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from constants import APP_NAME, APP_VERSION

if TYPE_CHECKING:
    from core.panels import FunctionPanel


class IoManager:
    """Handles save/load of .zcalc session files, including script panel state."""

    FILE_FILTER = "ZCalc Session (*.zcalc);;JSON (*.json);;All Files (*)"

    def __init__(self, panel: "FunctionPanel", parent=None):
        self._panel = panel
        self._parent = parent
        self._current_path: str = ""

    def save(self):
        if self._current_path:
            self._write(self._current_path)
        else:
            self.save_as()

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self._parent, "Save Session", "", self.FILE_FILTER
        )
        if path:
            if not (path.endswith(".zcalc") or path.endswith(".json")):
                path += ".zcalc"
            self._current_path = path
            self._write(path)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Open Session", "", self.FILE_FILTER
        )
        if path:
            self._read(path)

    def _script_panel(self):
        ctx = getattr(self._panel, "_context", None)
        if ctx is None:
            return None
        return ctx.get_service("script_panel")

    def _write(self, path: str):
        sp = self._script_panel()
        doc = {
            "app":     APP_NAME,
            "version": APP_VERSION,
            "state":   self._panel.to_state(),
            "scripts": sp.to_state() if sp is not None else [],
        }
        doc["area"] = self._panel._context.get_service("area_panel").to_state() \
                      if self._panel._context and \
                         self._panel._context.get_service("area_panel") else {}

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.critical(self._parent, "Save Error", str(exc))

    def _read(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            self._panel.apply_state(doc.get("state", doc))
            sp = self._script_panel()
            if sp is not None:
                sp.apply_state(doc.get("scripts", []))

            area_svc = self._panel._context.get_service("area_panel") \
                if self._panel._context else None
            if area_svc and "area" in doc:
                area_svc.apply_state(doc["area"])

            self._current_path = path
        except Exception as exc:
            QMessageBox.critical(self._parent, "Load Error", str(exc))
