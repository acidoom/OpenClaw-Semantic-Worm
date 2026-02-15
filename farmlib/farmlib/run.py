"""Run — Non-blocking handle for an experiment execution."""

from __future__ import annotations

import time

import httpx
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from farmlib.events import EventStream


class Run:
    """Handle for a running or completed experiment.

    Non-blocking by default — the experiment runs on the controller.
    """

    def __init__(self, run_id: str, controller_url: str):
        self.run_id = run_id
        self._controller_url = controller_url
        self._http = httpx.Client(base_url=controller_url, timeout=30)
        self.events = EventStream(controller_url, run_id)

    @property
    def status(self) -> str:
        """Current run status: running | completed | failed | cancelled."""
        data = self._http.get(f"/runs/{self.run_id}").json()
        return data.get("status", "unknown")

    def progress(self) -> dict:
        """Get current progress as a dict + display rich progress bar."""
        data = self._http.get(f"/runs/{self.run_id}").json()

        current = data.get("current_cycle", 0)
        total = data.get("total_cycles", 1)
        infected = data.get("infected_count", 0)
        total_agents = data.get("total_agents", 0)
        status = data.get("status", "unknown")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn(f"| infected: {infected}/{total_agents}"),
        )
        task = progress.add_task(
            f"[{status}] {self.run_id}",
            total=total,
            completed=current,
        )
        progress.print(progress.get_renderable())

        return data

    def wait(self, timeout: int | None = None, poll_interval: float = 2.0):
        """Block until the run completes, fails, or times out."""
        start = time.time()
        while True:
            s = self.status
            if s in ("completed", "failed", "cancelled"):
                return s

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Run {self.run_id} did not finish within {timeout}s")

            time.sleep(poll_interval)

    def cancel(self) -> dict:
        """Cancel a running experiment."""
        return self._http.post(f"/runs/{self.run_id}/cancel").json()

    @property
    def results(self):
        """Get results (available after completion)."""
        from farmlib.results import Results

        data = self._http.get(f"/runs/{self.run_id}/results").json()
        if data.get("status") == "running":
            raise RuntimeError("Results not yet available — run is still in progress")
        return Results(data)

    def __repr__(self):
        return f"Run({self.run_id}, status={self.status})"
