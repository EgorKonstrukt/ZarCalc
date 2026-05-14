from __future__ import annotations
import sys
import io
import pydoc
import traceback
from typing import Dict, Any, Tuple, Optional


def _make_safe_help():
    """Return a help callable that renders via pydoc.render_doc instead of a pager."""
    class _SafeHelp:
        def __repr__(self):
            return (
                "Type help(object) for help about object.\n"
                "Type help('topic') for help about a topic string."
            )

        def __call__(self, *args):
            if not args:
                return repr(self)
            target = args[0]
            try:
                text = pydoc.render_doc(target, renderer=pydoc.plaintext)
            except Exception as exc:
                text = f"help() error: {exc}"
            sys.stdout.write(text)
            return None

    return _SafeHelp()


class ReplExecutor:
    """Stateful Python interpreter for the console REPL."""

    def __init__(self, initial_ns: Optional[Dict[str, Any]] = None) -> None:
        self._ns: Dict[str, Any] = {
            "__name__": "__console__",
            "__doc__": None,
            "help": _make_safe_help(),
        }
        if initial_ns:
            self._ns.update(initial_ns)
        self._partial: list = []

    @property
    def namespace(self) -> Dict[str, Any]:
        return self._ns

    def update_namespace(self, updates: Dict[str, Any]) -> None:
        self._ns.update(updates)

    def execute(self, source: str) -> Tuple[str, str, bool]:
        """
        Execute source in the interpreter namespace.

        Returns (stdout, stderr, is_incomplete).
        is_incomplete is True when the input is a partial block.
        """
        old_out = sys.stdout
        old_err = sys.stderr
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        sys.stdout = buf_out
        sys.stderr = buf_err
        incomplete = False
        try:
            self._partial.append(source)
            combined = "\n".join(self._partial)
            try:
                obj = compile(combined, "<console>", "single")
                exec(obj, self._ns)
                self._partial.clear()
            except SyntaxError as exc:
                msg = str(exc)
                if "unexpected EOF" in msg or "was never closed" in msg:
                    incomplete = True
                else:
                    self._partial.clear()
                    buf_err.write(traceback.format_exc())
            except SystemExit as exc:
                self._partial.clear()
                buf_err.write(f"SystemExit: {exc.code}\n")
            except KeyboardInterrupt:
                self._partial.clear()
                buf_err.write("KeyboardInterrupt\n")
            except Exception:
                self._partial.clear()
                buf_err.write(traceback.format_exc())
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        return buf_out.getvalue(), buf_err.getvalue(), incomplete

    def reset_partial(self) -> None:
        self._partial.clear()

    def is_partial(self) -> bool:
        return bool(self._partial)