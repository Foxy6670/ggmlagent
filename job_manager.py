"""
Background job manager for long-running or potentially-hanging commands.

Jobs are daemon threads — they die automatically when the main process exits.
The agent interacts with them via /bg, /fg, and /jobs commands, which are
intercepted in agent.py before reaching the CommandDispatcher.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    id: int
    command: str
    result: str = ""
    started_at: float = field(default_factory=time.monotonic)
    _done: threading.Event = field(default_factory=threading.Event)

    def wait(self, timeout: float) -> str | None:
        """Block up to *timeout* seconds. Returns result string, or None on timeout."""
        if self._done.wait(timeout):
            return self.result
        return None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def elapsed(self) -> int:
        return int(time.monotonic() - self.started_at)


class JobManager:
    def __init__(self):
        self._jobs: dict[int, Job] = {}
        self._next_id = 1

    def start(self, command: str, fn: Callable[[], str]) -> Job:
        """Launch *fn* in a daemon thread and return the associated Job."""
        job = Job(id=self._next_id, command=command)
        self._next_id += 1
        self._jobs[job.id] = job

        def _run():
            try:
                job.result = fn() or ""
            except Exception as e:
                job.result = f"[error] {type(e).__name__}: {e}"
            finally:
                job._done.set()

        threading.Thread(target=_run, daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def list_all(self) -> str:
        if not self._jobs:
            return "No background jobs."
        lines = []
        for j in sorted(self._jobs.values(), key=lambda x: x.id):
            status = "done" if j.done else f"running {j.elapsed()}s"
            lines.append(f"  #{j.id}  [{status:>14}]  {j.command[:70]}")
        return "Background jobs:\n" + "\n".join(lines)
