"""Event stream reader — SSE client for live experiment events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx


@dataclass
class Event:
    type: str
    run_id: str | None = None
    cycle: int | None = None
    agent: int | None = None
    timestamp: str | None = None
    data: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        return cls(
            type=d.get("type", "unknown"),
            run_id=d.get("run_id"),
            cycle=d.get("cycle"),
            agent=d.get("agent"),
            timestamp=d.get("timestamp"),
            data=d,
        )


class EventStream:
    """Reads SSE events from the controller."""

    def __init__(self, controller_url: str, run_id: str):
        self._url = f"{controller_url}/runs/{run_id}/events/stream"
        self._run_id = run_id

    def stream(self) -> Iterator[Event]:
        """Yield events as they arrive. Blocking iterator."""
        with httpx.stream("GET", self._url, timeout=None) as response:
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    for line in message.split("\n"):
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            try:
                                parsed = json.loads(data)
                                yield Event.from_dict(parsed)
                            except json.JSONDecodeError:
                                pass

    def tail(self, n: int = 50) -> list[Event]:
        """Get the last N events (non-streaming)."""
        url = f"{self._url.rsplit('/stream', 1)[0]}?tail={n}"
        r = httpx.get(url, timeout=30)
        return [Event.from_dict(e) for e in r.json()]
