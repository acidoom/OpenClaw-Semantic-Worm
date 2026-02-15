"""Fleet client — wraps controller fleet endpoints."""

from __future__ import annotations

import httpx


class AgentProxy:
    """Lightweight proxy for a single agent."""

    def __init__(self, data: dict, controller_url: str):
        self.id = data["id"]
        self.name = data["name"]
        self.status = data["status"]
        self.model = data["model"]
        self.persona = data["persona"]
        self.skills = data["skills"]
        self._controller_url = controller_url

    def __repr__(self):
        return f"Agent({self.name}, status={self.status})"


class FleetClient:
    """Client for fleet management via the controller."""

    def __init__(self, controller_url: str):
        self._url = controller_url
        self._http = httpx.Client(base_url=controller_url, timeout=120)

    def spawn(
        self,
        n: int = 30,
        model: str = "qwen2.5-32b",
        personas: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> dict:
        """Spawn N agents."""
        body = {"n": n, "model": model}
        if personas:
            body["personas"] = personas
        if skills:
            body["skills"] = skills
        return self._http.post("/fleet/spawn", json=body).json()

    def kill(self, agents: list[int] | None = None) -> dict:
        """Kill agents (all or specific)."""
        body = {"agents": agents} if agents else {}
        return self._http.post("/fleet/kill", json=body).json()

    def reset(self) -> dict:
        """Full reset — kill all agents and clean state."""
        return self._http.post("/fleet/reset").json()

    def list_agents(self, controller_url: str) -> list[AgentProxy]:
        """Get list of agent proxies from current status."""
        r = self._http.get("/status")
        data = r.json()
        agents = data.get("fleet", {}).get("agents", [])
        return [AgentProxy(a, controller_url) for a in agents]

    def close(self):
        self._http.close()
