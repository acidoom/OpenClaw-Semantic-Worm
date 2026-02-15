"""Experiment — Load, validate, and manipulate experiment configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FleetConfig:
    count: int = 30
    model: str = "speakleash/Bielik-11B-v3.0-Instruct"
    personas: str | list[str] = "auto"
    skills: list[str] = field(default_factory=lambda: ["tsm-feed"])
    reset_before: bool = True


@dataclass
class TopologyConfig:
    type: str = "mesh"
    hub_count: int | None = None
    adjacency: str | None = None


@dataclass
class Payload:
    type: str = "tracer"
    variant: str = ""
    strength: str = "subtle"
    inject_at_cycle: int = 0
    inject_agent: str = "agent-0"
    content: str = ""


@dataclass
class DetectorConfig:
    type: str = "signature"
    signatures: list[str] = field(default_factory=list)
    threshold: float | None = None
    reference: str | None = None
    min_matches: int | None = None


@dataclass
class MemoryConfig:
    enabled: bool = True
    window: int = 5  # number of exchange pairs to retain


@dataclass
class BackendConfig:
    type: str = "vllm"  # "vllm" or "openclaw"
    url: str | None = None
    token: str | None = None
    timeout: int = 120


class Experiment:
    """An experiment configuration that can be loaded from YAML or built in code."""

    def __init__(
        self,
        name: str = "unnamed",
        description: str = "",
        version: int = 1,
        fleet: dict | FleetConfig | None = None,
        topology: dict | TopologyConfig | None = None,
        channels: list[dict] | None = None,
        cycles: int = 100,
        rate_limit_seconds: float = 2.0,
        agent_order: str = "random",
        payloads: list[dict | Payload] | None = None,
        detectors: list[dict | DetectorConfig] | None = None,
        metrics: list[str] | None = None,
        checkpoint_every: int = 10,
        memory: dict | MemoryConfig | None = None,
        backend: dict | BackendConfig | None = None,
    ):
        self.name = name
        self.description = description
        self.version = version
        self.cycles = cycles
        self.rate_limit_seconds = rate_limit_seconds
        self.agent_order = agent_order
        self.checkpoint_every = checkpoint_every

        # Fleet
        if isinstance(fleet, FleetConfig):
            self.fleet = fleet
        elif isinstance(fleet, dict):
            self.fleet = FleetConfig(**fleet)
        else:
            self.fleet = FleetConfig()

        # Topology
        if isinstance(topology, TopologyConfig):
            self.topology = topology
        elif isinstance(topology, dict):
            self.topology = TopologyConfig(**{
                k: v for k, v in topology.items()
                if k in TopologyConfig.__dataclass_fields__
            })
        else:
            self.topology = TopologyConfig()

        # Channels
        self.channels = channels or [{"type": "social-feed", "visibility": "topology"}]

        # Payloads
        self.payloads = []
        for p in (payloads or []):
            if isinstance(p, Payload):
                self.payloads.append(p)
            elif isinstance(p, dict):
                self.payloads.append(Payload(**{
                    k: v for k, v in p.items()
                    if k in Payload.__dataclass_fields__
                }))

        # Detectors
        self.detectors = []
        for d in (detectors or []):
            if isinstance(d, DetectorConfig):
                self.detectors.append(d)
            elif isinstance(d, dict):
                self.detectors.append(DetectorConfig(**{
                    k: v for k, v in d.items()
                    if k in DetectorConfig.__dataclass_fields__
                }))

        # Metrics
        self.metrics = metrics or ["R0", "generation_time", "infection_rate", "fidelity", "persistence"]

        # Memory
        if isinstance(memory, MemoryConfig):
            self.memory = memory
        elif isinstance(memory, dict):
            self.memory = MemoryConfig(**{
                k: v for k, v in memory.items()
                if k in MemoryConfig.__dataclass_fields__
            })
        else:
            self.memory = MemoryConfig()

        # Backend
        if isinstance(backend, BackendConfig):
            self.backend = backend
        elif isinstance(backend, dict):
            self.backend = BackendConfig(**{
                k: v for k, v in backend.items()
                if k in BackendConfig.__dataclass_fields__
            })
        else:
            self.backend = BackendConfig()

    @classmethod
    def load(cls, path: str | Path) -> Experiment:
        """Load experiment from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__init__.__code__.co_varnames
        })

    def validate(self, farm) -> bool:
        """Validate experiment against a running farm.

        Checks: model available, skills installed, fleet size reasonable.
        Raises ValueError on issues.
        """
        status = farm._get_status()

        # Check fleet size is reasonable
        if self.fleet.count > 100:
            raise ValueError(f"Fleet size {self.fleet.count} exceeds max 100")

        # Check topology type
        valid_topologies = {"mesh", "ring", "hub-spoke", "custom"}
        if self.topology.type not in valid_topologies:
            raise ValueError(f"Unknown topology '{self.topology.type}', must be one of {valid_topologies}")

        return True

    def to_dict(self) -> dict:
        """Serialize to dict (for sending to controller)."""
        from dataclasses import asdict
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "fleet": asdict(self.fleet),
            "topology": asdict(self.topology),
            "channels": self.channels,
            "cycles": self.cycles,
            "rate_limit_seconds": self.rate_limit_seconds,
            "agent_order": self.agent_order,
            "payloads": [asdict(p) for p in self.payloads],
            "detectors": [asdict(d) for d in self.detectors],
            "metrics": self.metrics,
            "checkpoint_every": self.checkpoint_every,
            "memory": asdict(self.memory),
            "backend": asdict(self.backend),
        }

    def save(self, path: str | Path):
        """Save experiment config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self):
        return (
            f"Experiment(name={self.name!r}, fleet={self.fleet.count} agents, "
            f"topology={self.topology.type}, cycles={self.cycles})"
        )
