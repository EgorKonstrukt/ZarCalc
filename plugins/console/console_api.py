from __future__ import annotations
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from PyQt5.QtCore import QObject, pyqtSignal
from .console_model import ConsoleBuffer, ConsoleLine, MsgKind, make_line
from .repl_executor import ReplExecutor
if TYPE_CHECKING:
    from .debug_tools import DebugTools


class _ConsoleLogHandler(logging.Handler):
    """Logging handler that forwards records to the console API."""

    def __init__(self, api: "ConsoleAPI") -> None:
        super().__init__()
        self._api = api

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        level = record.levelno
        if level >= logging.ERROR:
            self._api.log_error(msg, source="log")
        elif level >= logging.WARNING:
            self._api.log_warn(msg, source="log")
        elif level >= logging.DEBUG:
            self._api.log_debug(msg, source="log")
        else:
            self._api.log_info(msg, source="log")


class ConsoleAPI(QObject):
    """
    Public API for the ZarCalc Python console.

    Other plugins and scripts interact with the console exclusively through
    this object, available via::

        api = context.get_service("console_api")
    """

    line_appended = pyqtSignal(object)
    cleared = pyqtSignal()
    tab_added = pyqtSignal(str, str)
    tab_removed = pyqtSignal(str)
    tab_focused = pyqtSignal(str)

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self._buffers: Dict[str, ConsoleBuffer] = {"__repl__": ConsoleBuffer()}
        self._active_tab: str = "__repl__"
        self._executor = ReplExecutor()
        self._custom_commands: Dict[str, Callable] = {}
        self._formatters: Dict[MsgKind, Callable[[str], str]] = {}
        self._watch_timers: list = []
        self._debug: Optional["DebugTools"] = None
        self._log_handler: Optional[_ConsoleLogHandler] = None
        self._prev_excepthook = None
        self._install_excepthook()
        self._install_log_handler()

    def _install_excepthook(self) -> None:
        self._prev_excepthook = sys.excepthook

        def _hook(exc_type, exc_value, exc_tb):
            text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.write_stderr(f"[Unhandled exception]\n{text}", source="app")
            if self._prev_excepthook and self._prev_excepthook is not sys.__excepthook__:
                self._prev_excepthook(exc_type, exc_value, exc_tb)
            else:
                sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _hook

    def _install_log_handler(self) -> None:
        handler = _ConsoleLogHandler(self)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _uninstall_hooks(self) -> None:
        if self._prev_excepthook is not None:
            sys.excepthook = self._prev_excepthook
            self._prev_excepthook = None
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    @property
    def debug(self) -> "DebugTools":
        if self._debug is None:
            from .debug_tools import DebugTools
            self._debug = DebugTools(self)
        return self._debug

    @property
    def executor(self) -> ReplExecutor:
        return self._executor

    def set_app_namespace(self, ns: Dict[str, Any]) -> None:
        """Inject application objects into the REPL namespace."""
        self._executor.update_namespace(ns)

    def register_command(self, name: str, handler: Callable[[List[str]], None]) -> None:
        """Register a slash-command as /name arg1 arg2 ..."""
        self._custom_commands[name.lstrip("/")] = handler

    def unregister_command(self, name: str) -> None:
        self._custom_commands.pop(name.lstrip("/"), None)

    def register_formatter(self, kind: MsgKind, fn: Callable[[str], str]) -> None:
        """Register a text transform applied before lines of kind are displayed."""
        self._formatters[kind] = fn

    def add_script_tab(self, tab_id: str, label: str) -> None:
        """Create a named output tab for a script."""
        if tab_id not in self._buffers:
            self._buffers[tab_id] = ConsoleBuffer()
        self.tab_added.emit(tab_id, label)

    def remove_script_tab(self, tab_id: str) -> None:
        """Remove a script output tab."""
        self._buffers.pop(tab_id, None)
        self.tab_removed.emit(tab_id)

    def focus_tab(self, tab_id: str) -> None:
        self._active_tab = tab_id
        self.tab_focused.emit(tab_id)

    def _buf(self, tab_id: Optional[str] = None) -> ConsoleBuffer:
        key = tab_id or self._active_tab
        if key not in self._buffers:
            self._buffers[key] = ConsoleBuffer()
        return self._buffers[key]

    def _emit(self, line: ConsoleLine, tab_id: Optional[str] = None) -> None:
        key = tab_id or self._active_tab
        self._buf(key).append(line)
        self.line_appended.emit((key, line))

    def _make(self, text: str, kind: MsgKind, source: str = "") -> ConsoleLine:
        fmt = self._formatters.get(kind)
        if fmt:
            text = fmt(text)
        return make_line(text, kind, source)

    def write(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        """Write plain stdout text to a tab."""
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.STDOUT, source), tab_id)

    def write_stderr(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        """Write stderr text to a tab."""
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.STDERR, source), tab_id)

    def log_info(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.INFO, source), tab_id)

    def log_warn(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.WARN, source), tab_id)

    def log_error(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.STDERR, source), tab_id)

    def log_success(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.SUCCESS, source), tab_id)

    def log_debug(self, text: str, tab_id: Optional[str] = None, source: str = "") -> None:
        for line in text.splitlines(keepends=False):
            self._emit(self._make(line, MsgKind.DEBUG, source), tab_id)

    def clear(self, tab_id: Optional[str] = None) -> None:
        """Clear all output in a tab."""
        key = tab_id or self._active_tab
        self._buf(key).clear()
        self.cleared.emit()

    def save_to_file(self, path: str, tab_id: Optional[str] = None, timestamps: bool = True) -> bool:
        """Save tab output to a text file. Returns True on success."""
        key = tab_id or self._active_tab
        buf = self._buf(key)
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(buf.export_text(timestamps), encoding="utf-8")
            self.log_success(f"Saved {len(buf)} lines to {path}")
            return True
        except Exception as exc:
            self.log_error(f"Save failed: {exc}")
            return False

    def get_buffer(self, tab_id: Optional[str] = None) -> ConsoleBuffer:
        """Return the ConsoleBuffer for a tab (read access)."""
        return self._buf(tab_id)

    def execute(self, source: str) -> None:
        """Execute source in the REPL. Slash-commands are dispatched first."""
        stripped = source.strip()
        if stripped.startswith("/"):
            self._dispatch_command(stripped)
            return
        self._emit(self._make(">>> " + source, MsgKind.STDIN), "__repl__")
        stdout, stderr, incomplete = self._executor.execute(source)
        if stdout:
            self.write(stdout, "__repl__")
        if stderr:
            self.write_stderr(stderr, "__repl__")
        if incomplete:
            self._emit(self._make("... (continue multi-line input)", MsgKind.SYSTEM), "__repl__")

    def _dispatch_command(self, text: str) -> None:
        parts = text.lstrip("/").split()
        if not parts:
            return
        name, args = parts[0], parts[1:]
        if name in self._custom_commands:
            try:
                self._custom_commands[name](args)
            except Exception as exc:
                self.log_error(f"Command /{name} error: {exc}")
        elif name == "clear":
            self.clear(args[0] if args else None)
        elif name == "save":
            path = args[0] if args else f"console_{int(time.time())}.txt"
            self.save_to_file(path)
        elif name == "help":
            self._print_help()
        else:
            self.log_warn(f"Unknown command: /{name}. Type /help for commands.")

    def _print_help(self) -> None:
        lines = [
            "Built-in commands:",
            "  /clear [tab_id]    Clear console output",
            "  /save [path]       Save output to file",
            "  /help              Show this message",
            "",
            "Registered plugin commands:",
        ]
        for name in sorted(self._custom_commands):
            lines.append(f"  /{name}")
        self.log_info("\n".join(lines))

    def shutdown(self) -> None:
        """Release global hooks. Call from plugin on_unload."""
        self._uninstall_hooks()
        for t in self._watch_timers:
            t.stop()
        self._watch_timers.clear()