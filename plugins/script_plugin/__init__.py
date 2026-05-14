from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

from core.plugins.plugin_base import PanelPlugin, PluginMeta

if TYPE_CHECKING:
    from core import AppContext
    from PyQt5.QtWidgets import QWidget

PLUGIN_META = PluginMeta(
    id="zarcalc.script",
    name="Script",
    version="1.1.0",
    author="ZarCalc",
    description="Python script item with chart/animation integration and profiler.",
    dependencies=[],
)


def _find_panel(context: "AppContext"):
    """Return the ScriptPanel sidebar widget, or None if not present."""
    mw = context.main_window
    sa = mw.findChild(type(None).__class__, "sidebar_area") if mw else None
    sidebar = getattr(mw, "_bottom_area", None)
    if sidebar is None:
        return None
    from .script_panel import ScriptPanel
    for child in sidebar.findChildren(ScriptPanel):
        return child
    return None


def _find_rows(context: "AppContext"):
    """Return all active ScriptRow widgets."""
    panel = _find_panel(context)
    if panel is None:
        return []
    return list(panel._script_rows)


def _row_by_name(context: "AppContext", name: str):
    """Find a ScriptRow whose stem or filename matches name (case-insensitive)."""
    name_l = name.lower()
    for row in _find_rows(context):
        if row._script_path:
            from pathlib import Path
            p = Path(row._script_path)
            if p.stem.lower() == name_l or p.name.lower() == name_l:
                return row
    return None


def _register_commands(context: "AppContext") -> None:
    """Register all /script* and utility slash-commands into ConsoleAPI."""
    api = context.get_service("console_api")
    if api is None:
        return

    def _cmd_scripts(args: List[str]) -> None:
        """List all loaded scripts and their status."""
        rows = _find_rows(context)
        if not rows:
            api.log_info("No scripts loaded.")
            return
        lines = ["Scripts:"]
        for row in rows:
            name = row._script_name() or "(unnamed)"
            state = "running" if row._running else "stopped"
            path = row._script_path or "(no file)"
            lines.append(f"  [{state:7}]  {name}  —  {path}")
        api.log_info("\n".join(lines))

    def _cmd_run(args: List[str]) -> None:
        """Run a script by name. Usage: /run <name>"""
        if not args:
            api.log_warn("Usage: /run <script_name>")
            return
        row = _row_by_name(context, args[0])
        if row is None:
            api.log_warn(f"Script not found: {args[0]}")
            return
        row._on_run()
        api.log_info(f"Started: {row._script_name()}")

    def _cmd_stop(args: List[str]) -> None:
        """Stop a script by name, or all scripts. Usage: /stop <name|all>"""
        if not args or args[0].lower() == "all":
            rows = _find_rows(context)
            for row in rows:
                if row._running:
                    row._on_stop()
            api.log_info(f"Stopped {len(rows)} script(s).")
            return
        row = _row_by_name(context, args[0])
        if row is None:
            api.log_warn(f"Script not found: {args[0]}")
            return
        row._on_stop()
        api.log_info(f"Stopped: {row._script_name()}")

    def _cmd_reload(args: List[str]) -> None:
        """Reload (stop + run) a script by name. Usage: /reload <name|all>"""
        if not args or args[0].lower() == "all":
            rows = [r for r in _find_rows(context) if r._running]
            for row in rows:
                row._on_stop()
                row._on_run()
            api.log_info(f"Reloaded {len(rows)} running script(s).")
            return
        row = _row_by_name(context, args[0])
        if row is None:
            api.log_warn(f"Script not found: {args[0]}")
            return
        row._on_stop()
        row._on_run()
        api.log_info(f"Reloaded: {row._script_name()}")

    def _cmd_prof(args: List[str]) -> None:
        """Show profiler summary for a script. Usage: /prof <name>"""
        if not args:
            api.log_warn("Usage: /prof <script_name>")
            return
        row = _row_by_name(context, args[0])
        if row is None:
            api.log_warn(f"Script not found: {args[0]}")
            return
        s = row._profiler.summary()
        lines = [
            f"Profiler — {row._script_name()}",
            f"  wall time : {s['wall_s']:.3f}s",
            f"  CPU time  : {s['cpu_s']*1000:.1f}ms",
            f"  RAM now   : {s['mem_mb']:.1f} MB",
            f"  RAM delta : {s['mem_delta_mb']:+.1f} MB",
            f"  RAM peak  : {s['mem_peak_mb']:.1f} MB",
            f"  CPU%      : {s['cpu_pct']:.0f}%",
        ]
        api.log_info("\n".join(lines))

    def _cmd_ns(args: List[str]) -> None:
        """List REPL namespace names, or filter by prefix. Usage: /ns [prefix]"""
        ns = api.executor.namespace
        prefix = args[0].lower() if args else ""
        names = sorted(k for k in ns if not k.startswith("__") and k.lower().startswith(prefix))
        if not names:
            api.log_info("(empty namespace)")
            return
        cols = 4
        rows_text = []
        for i in range(0, len(names), cols):
            rows_text.append("  " + "  ".join(f"{n:<20}" for n in names[i:i+cols]))
        api.log_info(f"Namespace ({len(names)} names):\n" + "\n".join(rows_text))

    def _cmd_timeit(args: List[str]) -> None:
        """Time an expression N times. Usage: /timeit <expr> [n=100]"""
        if not args:
            api.log_warn("Usage: /timeit <expr> [n]")
            return
        expr = args[0]
        n = int(args[1]) if len(args) > 1 else 100
        api.debug.time_it(expr, n)

    def _cmd_mem(args: List[str]) -> None:
        """Print GC memory summary."""
        api.debug.memory_summary()

    def _cmd_gc(args: List[str]) -> None:
        """Force garbage collection."""
        api.debug.gc_collect()

    def _cmd_watch(args: List[str]) -> None:
        """Watch an expression every N seconds. Usage: /watch <expr> [interval_s]"""
        if not args:
            api.log_warn("Usage: /watch <expr> [interval_s]")
            return
        expr = args[0]
        interval = float(args[1]) if len(args) > 1 else 1.0
        api.debug.watch(expr, interval)

    def _cmd_unwatch(args: List[str]) -> None:
        """Stop all active watch() timers."""
        api.debug.stop_watches()

    def _cmd_inspect(args: List[str]) -> None:
        """Inspect an object from the REPL namespace. Usage: /inspect <name>"""
        if not args:
            api.log_warn("Usage: /inspect <name>")
            return
        ns = api.executor.namespace
        name = args[0]
        if name not in ns:
            api.log_warn(f"Name not found in namespace: {name}")
            return
        api.debug.inspect(ns[name])

    def _cmd_threads(args: List[str]) -> None:
        """Print all active thread stacks."""
        api.debug.traceback()

    def _cmd_viewport(args: List[str]) -> None:
        """Print current chart viewport."""
        chart = context.chart
        vp = chart.getViewBox() if hasattr(chart, "getViewBox") else None
        if vp is None:
            api.log_warn("Viewport not available.")
            return
        xr, yr = vp.viewRange()
        api.log_info(
            f"Viewport  x=[{xr[0]:.4g}, {xr[1]:.4g}]  y=[{yr[0]:.4g}, {yr[1]:.4g}]"
        )

    def _cmd_params(args: List[str]) -> None:
        """Print all current slider parameter values."""
        params = context.panel.get_params()
        if not params:
            api.log_info("No parameters defined.")
            return
        lines = ["Parameters:"]
        for k, v in sorted(params.items()):
            lines.append(f"  {k:<20} = {v}")
        api.log_info("\n".join(lines))

    def _cmd_autofit(args: List[str]) -> None:
        """Autofit the chart viewport."""
        chart = context.chart
        if hasattr(chart, "autofit"):
            chart.autofit()
            api.log_info("Autofit applied.")

    commands = {
        "scripts":   _cmd_scripts,
        "run":       _cmd_run,
        "stop":      _cmd_stop,
        "reload":    _cmd_reload,
        "prof":      _cmd_prof,
        "ns":        _cmd_ns,
        "timeit":    _cmd_timeit,
        "mem":       _cmd_mem,
        "gc":        _cmd_gc,
        "watch":     _cmd_watch,
        "unwatch":   _cmd_unwatch,
        "inspect":   _cmd_inspect,
        "threads":   _cmd_threads,
        "viewport":  _cmd_viewport,
        "params":    _cmd_params,
        "autofit":   _cmd_autofit,
    }
    for name, handler in commands.items():
        api.register_command(name, handler)


def _unregister_commands(context: "AppContext") -> None:
    api = context.get_service("console_api")
    if api is None:
        return
    for name in (
        "scripts", "run", "stop", "reload", "prof",
        "ns", "timeit", "mem", "gc", "watch", "unwatch",
        "inspect", "threads", "viewport", "params", "autofit",
    ):
        api.unregister_command(name)


class ScriptPlugin(PanelPlugin):
    """PanelPlugin that contributes a Script item type to the + Add menu."""

    meta = PLUGIN_META

    @property
    def menu_label(self) -> str:
        return "Script"

    def create_item(self, context: "AppContext") -> "QWidget":
        from .script_row import ScriptRow
        return ScriptRow(context)

    def to_item_state(self, widget: "QWidget") -> dict:
        return widget.to_state() if hasattr(widget, "to_state") else {}

    def restore_item(self, context: "AppContext", state: dict) -> "QWidget":
        from .script_row import ScriptRow
        row = ScriptRow(context)
        row.apply_state(state)
        return row

    def on_load(self, context: "AppContext") -> None:
        _register_commands(context)

    def on_unload(self, context: "AppContext") -> None:
        _unregister_commands(context)


def get_plugin() -> ScriptPlugin:
    return ScriptPlugin()