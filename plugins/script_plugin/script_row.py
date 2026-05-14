from __future__ import annotations
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import pyqtSignal, QTimer, QFileSystemWatcher, Qt

if TYPE_CHECKING:
    from core import AppContext
    from ..console.console_api import ConsoleAPI

from .script_api import ScriptAPI
from .script_runner import build_namespace, run_script, DEFAULT_TIMEOUT_S
from .editor_launcher import open_in_editor
from .profiler import ScriptProfiler, _PSUTIL

_FRAME_STYLE = (
    "QFrame#script_item{"
    "background:#f4fff4;"
    "border-left:4px solid #27ae60;"
    "border-top:1px solid #b2dfb2;"
    "border-right:1px solid #b2dfb2;"
    "border-bottom:1px solid #b2dfb2;"
    "margin:1px 0px;"
    "}"
)
_BTN = (
    "QPushButton{{background:{bg};color:white;border:none;border-radius:3px;"
    "font-size:10px;padding:2px 7px;}}"
    "QPushButton:hover{{background:{hv};}}"
    "QPushButton:disabled{{background:#ccc;color:#999;}}"
)
_BTN_RUN  = _BTN.format(bg="#27ae60", hv="#1e8449")
_BTN_STOP = _BTN.format(bg="#e74c3c", hv="#c0392b")
_BTN_EDIT = _BTN.format(bg="#2980b9", hv="#1a6fa0")
_BTN_LOAD = _BTN.format(bg="#7f8c8d", hv="#636e72")
_BTN_RM   = (
    "QPushButton{background:transparent;color:#aaa;border:none;"
    "font-size:13px;font-weight:bold;padding:0 4px;}"
    "QPushButton:hover{color:#e74c3c;}"
)
_ST_OK    = "QLabel{font-size:9px;color:#27ae60;}"
_ST_ERR   = "QLabel{font-size:9px;color:#e74c3c;}"
_ST_IDLE  = "QLabel{font-size:9px;color:#95a5a6;}"
_ST_RUN   = "QLabel{font-size:9px;color:#f39c12;}"
_PROF_STYLE = (
    "QLabel{font-size:9px;color:#555;font-family:monospace;"
    "background:#eefaee;border-top:1px solid #c3e6c3;padding:2px 6px;}"
)
_WATCH_DEBOUNCE_MS = 600
_PROF_UPDATE_MS    = 500


class _ConsoleStream(io.TextIOBase):
    """File-like object that routes writes to a ConsoleAPI tab."""

    def __init__(self, console_api: "ConsoleAPI", tab_id: str, is_err: bool) -> None:
        super().__init__()
        self._api = console_api
        self._tab_id = tab_id
        self._is_err = is_err

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._is_err:
            self._api.write_stderr(text, tab_id=self._tab_id)
        else:
            self._api.write(text, tab_id=self._tab_id)
        return len(text)

    def flush(self) -> None:
        pass


class ScriptRow(QFrame):
    """
    Panel item widget for one Python script file.

    Serialises as type='plugin_item' so FunctionPanel._restore_items
    can reconstruct it from a .zcalc session.
    Output is forwarded to the ConsoleAPI tab named after the script.
    """

    changed = pyqtSignal()
    removed = pyqtSignal(object)

    def __init__(self, context: "AppContext", script_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("script_item")
        self.setStyleSheet(_FRAME_STYLE)
        self._ctx = context
        self._script_path: Optional[str] = script_path
        self._running = False
        self._api: Optional[ScriptAPI] = None
        self._script_lines: Dict[str, Any] = {}
        self._profiler = ScriptProfiler()
        self._console_api: Optional["ConsoleAPI"] = None
        self._tab_id: Optional[str] = None
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_WATCH_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._auto_reload)
        self._prof_timer = QTimer(self)
        self._prof_timer.setInterval(_PROF_UPDATE_MS)
        self._prof_timer.timeout.connect(self._update_prof_label)
        self._build_ui()
        if script_path and os.path.isfile(script_path):
            self._watcher.addPath(script_path)

    def _get_console_api(self) -> Optional["ConsoleAPI"]:
        if self._console_api is None:
            self._console_api = self._ctx.get_service("console_api")
        return self._console_api

    def _tab_label(self) -> str:
        return self._display_name()

    def _ensure_tab(self) -> Optional[str]:
        """Create a console tab for this script and return its tab_id."""
        api = self._get_console_api()
        if api is None:
            return None
        tab_id = f"script::{id(self)}"
        api.add_script_tab(tab_id, self._tab_label())
        api.focus_tab(tab_id)
        self._tab_id = tab_id
        return tab_id

    def _remove_tab(self) -> None:
        api = self._get_console_api()
        if api is not None and self._tab_id is not None:
            api.remove_script_tab(self._tab_id)
        self._tab_id = None

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QWidget()
        top.setStyleSheet("background:transparent;")
        header = QHBoxLayout(top)
        header.setContentsMargins(6, 3, 4, 3)
        header.setSpacing(4)
        icon = QLabel(">>")
        icon.setStyleSheet("QLabel{color:#27ae60;font-size:13px;font-weight:bold;}")
        icon.setFixedWidth(20)
        self._name_lbl = QLabel(self._display_name())
        self._name_lbl.setStyleSheet("QLabel{font-size:11px;font-weight:bold;color:#1a5c1a;}")
        self._name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._status_lbl = QLabel("idle")
        self._status_lbl.setStyleSheet(_ST_IDLE)
        self._status_lbl.setFixedWidth(150)
        self._status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._run_btn  = self._mk_btn("Run",  _BTN_RUN,  self._on_run)
        self._stop_btn = self._mk_btn("Stop", _BTN_STOP, self._on_stop)
        self._stop_btn.setEnabled(False)
        self._edit_btn = self._mk_btn("Edit", _BTN_EDIT, self._on_edit)
        self._load_btn = self._mk_btn("Load", _BTN_LOAD, self._on_load)
        rm_btn = QPushButton("X")
        rm_btn.setStyleSheet(_BTN_RM)
        rm_btn.setFixedSize(20, 20)
        rm_btn.setToolTip("Remove script")
        rm_btn.clicked.connect(lambda: self.removed.emit(self))
        for w in (icon, self._name_lbl, self._status_lbl,
                  self._run_btn, self._stop_btn,
                  self._edit_btn, self._load_btn, rm_btn):
            header.addWidget(w)
        outer.addWidget(top)
        self._path_lbl = QLabel("No file loaded")
        self._path_lbl.setStyleSheet(
            "QLabel{font-size:9px;color:#aaa;padding:0 6px 2px 28px;}"
        )
        self._path_lbl.setWordWrap(True)
        outer.addWidget(self._path_lbl)
        self._prof_lbl = QLabel("")
        self._prof_lbl.setStyleSheet(_PROF_STYLE)
        self._prof_lbl.setVisible(False)
        outer.addWidget(self._prof_lbl)

    def _mk_btn(self, text: str, style: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setStyleSheet(style)
        btn.setFixedHeight(20)
        btn.clicked.connect(slot)
        return btn

    def _display_name(self) -> str:
        return Path(self._script_path).stem if self._script_path else "Script"

    def _script_name(self) -> str:
        return Path(self._script_path).name if self._script_path else "script"

    def _on_file_changed(self, _path: str):
        if self._running:
            self._debounce.start()

    def _auto_reload(self):
        if self._running and self._script_path and os.path.isfile(self._script_path):
            self._log_info("File changed - reloading script")
            self._on_stop()
            self._on_run()

    def _on_run(self):
        if not self._script_path or not os.path.isfile(self._script_path):
            self._set_status("No file loaded", error=True)
            return
        try:
            code = Path(self._script_path).read_text(encoding="utf-8")
        except Exception as exc:
            self._set_status(f"Read error: {exc}", error=True)
            self._log_error(f"Cannot read file: {exc}")
            return
        if self._api:
            self._api.cleanup()

        tab_id = self._ensure_tab()
        self._api = ScriptAPI(self._ctx, self)

        timeout = DEFAULT_TIMEOUT_S
        try:
            from config import Config
            timeout = float(Config().get("script_timeout_s") or DEFAULT_TIMEOUT_S)
        except Exception:
            pass

        self._profiler.start()

        stdout_real = sys.stdout
        stderr_real = sys.stderr
        console_api = self._get_console_api()
        if console_api is not None and tab_id is not None:
            sys.stdout = _ConsoleStream(console_api, tab_id, is_err=False)
            sys.stderr = _ConsoleStream(console_api, tab_id, is_err=True)

        ns = build_namespace(self._api)
        ok, err = run_script(code, ns, timeout_s=timeout)

        sys.stdout = stdout_real
        sys.stderr = stderr_real

        if ok:
            self._running = True
            self._run_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._set_status("running", ok=True)
            self._prof_lbl.setVisible(True)
            self._prof_timer.start()
            self._log_info(f"Script started (timeout={timeout:.0f}s)")
        else:
            if self._api:
                self._api.cleanup()
                self._api = None
            self._profiler.stop()
            self._remove_tab()
            lines = err.strip().splitlines()
            short = lines[-1] if lines else "Error"
            self._set_status(short, error=True)
            self._log_error(err)
            self._ctx.show_status(f"Script error: {short}", 6000)
            self._prof_lbl.setVisible(True)
            self._update_prof_label()
        self.changed.emit()

    def _on_stop(self):
        if self._api:
            self._api.cleanup()
            self._api = None
        if self._running:
            self._profiler.stop()
            self._log_info("Script stopped")
        self._remove_tab()
        self._prof_timer.stop()
        self._running = False
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._set_status("stopped")
        self._update_prof_label()
        self.changed.emit()

    def _update_prof_label(self):
        s = self._profiler.summary()
        parts = [
            f"wall {s['wall_s']:.1f}s",
            f"CPU {s['cpu_s']*1000:.0f}ms",
            f"RAM {s['mem_mb']:.1f}MB",
            f"D{s['mem_delta_mb']:+.1f}MB",
        ]
        if _PSUTIL and self._running:
            parts.append(f"CPU% {s['cpu_pct']:.0f}%")
        if not _PSUTIL:
            parts.append("(install psutil for CPU%/RAM)")
        self._prof_lbl.setText("  ".join(parts))

    def _on_edit(self):
        from config import Config
        editor_cmd = Config().get("script_editor") or ""
        if not self._script_path:
            self._create_new_and_edit(editor_cmd)
            return
        if not os.path.isfile(self._script_path):
            try:
                Path(self._script_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self._script_path).write_text("", encoding="utf-8")
            except Exception as exc:
                self._set_status(f"Cannot create: {exc}", error=True)
                return
        if not open_in_editor(self._script_path, editor_cmd):
            QMessageBox.warning(
                self, "Editor Not Found",
                f"Cannot launch '{editor_cmd}'.\n"
                "Change it in Settings > Script Editor.",
            )

    def _create_new_and_edit(self, editor_cmd: str):
        d = Path(__file__).parent.parent.parent / "scripts"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "new_script.py"
        n = 1
        while p.exists():
            p = d / f"new_script_{n}.py"
            n += 1
        p.write_text(
            '"""\nZCalc Script - use `api` to interact with the application.\n"""\n\n',
            encoding="utf-8",
        )
        self._set_path(str(p))
        if not open_in_editor(str(p), editor_cmd):
            QMessageBox.warning(
                self, "Editor Not Found",
                f"Cannot launch '{editor_cmd}'.\n"
                "Change it in Settings > Script Editor.",
            )

    def _on_load(self):
        d = Path(__file__).parent.parent.parent / "scripts"
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Script", str(d), "Python (*.py);;All Files (*)",
        )
        if path:
            self._set_path(path)
            self.changed.emit()

    def _set_path(self, path: str):
        for f in self._watcher.files():
            self._watcher.removePath(f)
        self._script_path = path
        self._name_lbl.setText(self._display_name())
        self._path_lbl.setText(path)
        if os.path.isfile(path):
            self._watcher.addPath(path)

    def _set_status(self, msg: str, error: bool = False, ok: bool = False, running: bool = False):
        self._status_lbl.setText(msg)
        if error:
            self._status_lbl.setStyleSheet(_ST_ERR)
        elif ok:
            self._status_lbl.setStyleSheet(_ST_OK)
        elif running:
            self._status_lbl.setStyleSheet(_ST_RUN)
        else:
            self._status_lbl.setStyleSheet(_ST_IDLE)

    def _log_info(self, msg: str):
        api = self._get_console_api()
        if api is not None:
            tab = self._tab_id or "__repl__"
            api.log_info(msg, tab_id=tab, source=self._script_name())

    def _log_error(self, msg: str):
        api = self._get_console_api()
        if api is not None:
            tab = self._tab_id or "__repl__"
            api.write_stderr(msg, tab_id=tab, source=self._script_name())

    def to_state(self) -> dict:
        return {
            "type":        "plugin_item",
            "plugin_id":   "zcalc.script",
            "script_path": self._script_path or "",
            "running":     self._running,
        }

    def apply_state(self, state: dict):
        path = state.get("script_path", "")
        if path and os.path.isfile(path):
            self._set_path(path)
        elif path:
            self._path_lbl.setText(f"(missing) {path}")
            self._script_path = path
            self._name_lbl.setText(self._display_name())
        if state.get("running", False) and self._script_path and os.path.isfile(self._script_path):
            self._on_run()

    def deleteLater(self):
        self._prof_timer.stop()
        if self._api:
            self._api.cleanup()
        if self._running:
            self._profiler.stop()
        self._remove_tab()
        super().deleteLater()