"""Farm — Main entry point for the SDK. Connect, manage fleet, run experiments."""

from __future__ import annotations

import httpx

from farmlib.config import CONTROLLER_URL
from farmlib.experiment import Experiment
from farmlib.feed import FeedClient
from farmlib.fleet import AgentProxy, FleetClient
from farmlib.run import Run
from farmlib.viz import agents_table, runs_table, status_table


class RunManager:
    """Access past and current runs."""

    def __init__(self, controller_url: str):
        self._url = controller_url
        self._http = httpx.Client(base_url=controller_url, timeout=30)

    def list(self) -> dict:
        """List all runs with their progress."""
        return self._http.get("/runs").json()

    def get(self, run_id: str) -> Run:
        """Get a Run handle by ID (reconnect after kernel restart)."""
        return Run(run_id, self._url)


class Farm:
    """Main SDK entry point. Connects to the farm controller."""

    def __init__(self, controller_url: str):
        self._controller_url = controller_url
        self._http = httpx.Client(base_url=controller_url, timeout=30)
        self._fleet_client = FleetClient(controller_url)
        self.feed = FeedClient()
        self.runs = RunManager(controller_url)

    @classmethod
    def connect(cls, controller: str = CONTROLLER_URL) -> Farm:
        """Connect to a running farm controller.

        Usage:
            farm = Farm.connect()
            farm = Farm.connect("http://my-dgx:9000")
        """
        # Verify controller is up
        r = httpx.get(f"{controller}/health", timeout=10)
        r.raise_for_status()

        return cls(controller)

    # ----- Status -----

    def _get_status(self) -> dict:
        return self._http.get("/status").json()

    def status(self):
        """Display rich status table."""
        health = self._http.get("/health").json()
        full = self._get_status()

        status_table(
            health=health,
            fleet=full.get("fleet", {}),
            feed=full.get("feed", {}),
            runs=full.get("runs", {}),
        )

    # ----- Fleet -----

    def spawn(
        self,
        n: int = 30,
        model: str = "speakleash/Bielik-11B-v3.0-Instruct",
        personas: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> dict:
        """Spawn N agents."""
        return self._fleet_client.spawn(n=n, model=model, personas=personas, skills=skills)

    def kill(self, agents: list[int] | None = None) -> dict:
        """Kill agents (all or specific IDs)."""
        return self._fleet_client.kill(agents)

    def reset(self) -> dict:
        """Full reset — kill agents, clear feed, clean state."""
        result = self._fleet_client.reset()
        self.feed.clear()
        return result

    @property
    def agents(self) -> list[AgentProxy]:
        """List of agent proxy objects."""
        full = self._get_status()
        agent_data = full.get("fleet", {}).get("agents", [])
        return [AgentProxy(a, self._controller_url) for a in agent_data]

    def show_agents(self):
        """Display rich agent table."""
        full = self._get_status()
        agents_table(full.get("fleet", {}).get("agents", []))

    # ----- Execute -----

    def execute(self, experiment: Experiment) -> Run:
        """Execute an experiment. Returns a non-blocking Run handle.

        Usage:
            exp = Experiment.load("experiments/semantic-worm/baseline.yaml")
            run = farm.execute(exp)
            run.wait()
        """
        r = self._http.post("/execute", json={"experiment": experiment.to_dict()})
        r.raise_for_status()
        data = r.json()
        return Run(data["run_id"], self._controller_url)

    def show_runs(self):
        """Display rich runs table."""
        data = self.runs.list()
        runs_table(data)

    def __repr__(self):
        return f"Farm(controller={self._controller_url!r})"
