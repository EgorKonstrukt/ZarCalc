"""
True multi-core compute pool using separate Python processes.
Bypasses GIL completely. Each worker process has its own interpreter.
Uses round-robin dispatch and lightweight future tracking.
"""
from __future__ import annotations
import os
from multiprocessing import Process, Queue
from typing import Callable, Optional

_WORKERS = min(max(os.cpu_count() or 2, 2), 8)
_QUEUE_MAXSIZE = 256


class _PendingFuture:
    """Minimal future for a cross-process result."""
    __slots__ = ("_done", "_result", "_callbacks")

    def __init__(self):
        self._done = False
        self._result = None
        self._callbacks: list = []

    def set_result(self, result):
        self._done = True
        self._result = result
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception:
                pass

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._result

    def add_done_callback(self, fn: Callable):
        if self._done:
            fn(self._result)
        else:
            self._callbacks.append(fn)


class ComputePool:
    """Singleton multi-process pool. Each worker runs worker_proc.worker_loop."""
    _instance: Optional["ComputePool"] = None

    def __new__(cls) -> "ComputePool":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._init()
            cls._instance = inst
        return cls._instance

    def _init(self):
        from worker_proc import worker_loop
        self._n = _WORKERS
        self._task_queues: list = []
        self._result_queues: list = []
        self._procs: list = []
        self._rr = 0
        self._pending: dict = {}
        for _ in range(self._n):
            tq = Queue(maxsize=_QUEUE_MAXSIZE)
            rq = Queue(maxsize=_QUEUE_MAXSIZE)
            p = Process(target=worker_loop, args=(tq, rq), daemon=True)
            p.start()
            self._task_queues.append(tq)
            self._result_queues.append(rq)
            self._procs.append(p)

    def _dispatch(self, task_id: str, msg: tuple) -> _PendingFuture:
        fut = _PendingFuture()
        self._pending[task_id] = fut
        idx = self._rr % self._n
        self._rr += 1
        try:
            self._task_queues[idx].put_nowait(msg)
        except Exception:
            self._task_queues[idx].put(msg, block=True, timeout=2.0)
        return fut

    def submit_yfx(self, task_id: str, expr: str, x_arr, extra: dict) -> _PendingFuture:
        return self._dispatch(task_id, ("yfx", task_id, expr, x_arr, extra))

    def submit_polar(self, task_id: str, expr: str, x_arr, extra: dict) -> _PendingFuture:
        return self._dispatch(task_id, ("polar", task_id, expr, x_arr, extra))

    def submit_param(self, task_id: str, expr: str, expr2: str, t_arr, extra: dict) -> _PendingFuture:
        return self._dispatch(task_id, ("param", task_id, expr, expr2, t_arr, extra))

    def submit_deriv(self, task_id: str, expr: str, x_arr, extra: dict, h: float) -> _PendingFuture:
        return self._dispatch(task_id, ("deriv", task_id, expr, x_arr, extra, h))

    def submit_deriv2(self, task_id: str, expr: str, x_arr, extra: dict, h: float) -> _PendingFuture:
        return self._dispatch(task_id, ("deriv2", task_id, expr, x_arr, extra, h))

    def submit_integral(self, task_id: str, expr: str, x_arr, extra: dict) -> _PendingFuture:
        return self._dispatch(task_id, ("integral", task_id, expr, x_arr, extra))

    def drain_results(self) -> list:
        """Non-blocking drain of all result queues. Returns list of (task_id, xy_tuple)."""
        out = []
        for rq in self._result_queues:
            while not rq.empty():
                try:
                    status, payload = rq.get_nowait()
                    task_id, xy = payload
                    out.append((task_id, xy))
                    fut = self._pending.pop(task_id, None)
                    if fut is not None:
                        fut.set_result(xy)
                except Exception:
                    pass
        return out

    def shutdown(self) -> None:
        for tq in self._task_queues:
            try:
                tq.put(None, timeout=0.5)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        ComputePool._instance = None


def get_pool() -> ComputePool:
    return ComputePool()