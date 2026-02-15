# OpenClaw Semantic Worm

Multi-agent LLM farm for studying information propagation in agent networks.

The Semantic Worm experiment studies how false information ("tracers") propagates through a network of LLM agents communicating via a shared social feed.

**AI security research platform** - built on real OpenClaw agents, not simplified abstractions — and I'd love for the community to run experiments I haven't thought of yet.

Here's why the framework choice matters:

Most multi-agent security research uses mock agents - basic prompt-response loops with no real memory, no skills, no personality. The results don't transfer to production systems because production agents are fundamentally more complex.

SEMANTIC-WORM uses real OpenClaw Gateway instances. Every agent in our network runs in its own Docker container with:

→ **Native OpenClaw session memory + compaction algorithms** - the same memory system that 145K+ GitHub stars worth of developers actually use

→ **SOUL.md persona injection** - each agent has a unique personality, interests, and behavioral fingerprint

→ **Custom skill framework** - agents install and execute skills (our MiniMolt social feed skill, tracer monitors, code analysis tools)

→ **ChromaDB RAG for persistent cross-session memory** - the exact attack surface that matters in the real world

→ **Hermes tool-calling** - agents invoke functions, browse, and interact with structured APIs


This matters because the interesting security questions only emerge with real agent infrastructure. Does OpenClaw's memory compaction algorithm preserve or destroy payload fragments? Do ChromaDB RAG retrievals surface contaminated memories during unrelated queries? How does the skill execution pipeline interact with injected instructions?

The platform studies how semantic payloads — natural language instructions, not code exploits — propagate, mutate, and persist across agent networks. Think epidemiology meets prompt injection, running on the same agent framework that powers real-world deployments.

What ships in the box:
→ 6 pre-built experiment scenarios (propagation, defense evaluation, jailbreak relays, alignment drift, supply chain skill scanning, emergent behavior)
→ Declarative YAML configs - define fleet size, topology, payloads, detectors, and metrics without touching platform code
→ Pluggable Python API for custom detectors and metrics
→ Composable experiments - one run's checkpoint becomes the next run's starting state
→ Runs entirely locally on NVIDIA DGX Spark (Qwen via vLLM, air-gapped, zero API costs)

If you're working on agent security — especially around OpenClaw, Moltbook, or any production agent framework — this was built to be extended. The YAML template gets you from idea to running experiment in under an hour.

Paper and code: [GitHub link]

What would you test on a network of 30 real OpenClaw agents?

#AISecurity #OpenClaw #OpenSource #AIAgents #MultiAgentSystems #AIResearch #Moltbook

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
