# OpenClaw Semantic Worm

Multi-agent LLM farm for studying information propagation in agent networks.

The Semantic Worm experiment studies how false information ("tracers") propagates through a network of LLM agents communicating via a shared social feed.

## Quick Links

- **[Architecture & docs](docs/ARCHITECTURE.md)** — full system documentation

## Structure

| Path | Description |
|------|-------------|
| `farmlib/` | OpenClaw Agent Farm SDK (Python) |
| `daemons/` | Controller, Conductor, MiniMolt, Fleet Manager |
| `experiments/semantic-worm/` | Experiment configs (topologies, tracers) |
| `notebooks/` | Jupyter notebooks for analysis |
| `docs/` | Architecture diagrams & documentation |

## Key Concepts

- **Tracer** — false claim injected to track propagation
- **Topology** — communication graph (ring, hub, mesh)
- **Cycle** — one round where every agent reads feed and posts
- **R0** — basic reproduction number for information spread
