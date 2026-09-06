"""Bounded background disk I/O; telemetry failures never escape control callbacks."""
import copy
import os
from pathlib import Path
import queue
import tempfile
import threading
import yaml


class BackgroundIO:
    def __init__(self, capacity=128):
        self.queue = queue.Queue(maxsize=capacity)
        self.error = ''
        self.dropped = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='robot-telemetry', daemon=True)
        self.thread.start()

    def submit(self, function, *args):
        if self.stopping.is_set():
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait((function, copy.deepcopy(args)))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                function, args = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                function(*args)
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
            finally:
                self.queue.task_done()

    def close(self, timeout=2.0):
        self.stopping.set()
        self.thread.join(timeout=timeout)


def save_model(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            yaml.safe_dump(state, file)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
