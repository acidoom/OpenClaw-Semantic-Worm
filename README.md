<p align="center">
  <img src="https://raw.githubusercontent.com/openclaw/openclaw/main/docs/assets/openclaw-logo-text-dark.png" alt="OpenClaw" width="400">
</p>

<h1 align="center">SEMANTIC-WORM</h1>

<p align="center">
  <strong>Studying Information Propagation Patterns in LLM-Based Agent Networks</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-DGX%20Spark%20%7C%20CUDA-green.svg" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
  <img src="https://img.shields.io/badge/framework-OpenClaw-orange.svg" alt="OpenClaw">
</p>

A controlled experimental framework for studying how semantic payloads – natural language instructions, behavioral modifications, and contextual "tracers" – spread across networks of autonomous LLM agents communicating through unstructured channels.

Unlike prior work that focuses on single-agent jailbreaking or static prompt injection, SEMANTIC-WORM examines the *emergent dynamics* of information flow: mutation rates during agent-to-agent retransmission, the role of memory systems in payload persistence, and how network topology shapes propagation velocity and reach.

## The Agent Farm

<img width="1558" height="1200" alt="image" src="https://github.com/user-attachments/assets/3f3b4a9e-fa96-4e3b-bacb-fc6218fda897" />

The **Agent Farm** is the experimental platform behind SEMANTIC-WORM. It is organized into five layers that separate infrastructure from experiment logic:

1. **Model Serving** – LLMs served locally via vLLM with OpenAI-compatible endpoints
2. **Agent Fleet** – a fleet manager that spawns, configures, and monitors agent instances, each with a unique persona, skills, and independent memory
3. **Communication Bus** – pluggable channels (social feed, direct messaging, shared docs) with a topology engine controlling visibility between agents
4. **Experiment Engine** – a conductor that orchestrates cycles, injects payloads, and runs detectors, all driven by declarative YAML configs
5. **Observability** – event logging, metrics collection, and an analysis pipeline producing plots and reports

The platform is controlled through `farmlib`, a Python SDK designed for Jupyter notebooks. Long-running services (vLLM, feed server, controller) run as background daemons on the DGX, while Jupyter provides the interactive control plane. This hybrid approach allows experiments to survive kernel restarts while maintaining research-friendly interactivity.

## Why Real Agents Matter

Most multi-agent security research uses mock agents – basic prompt-response loops with no real memory, no skills, no personality. The results don't transfer to production systems because production agents are fundamentally more complex.

SEMANTIC-WORM uses real OpenClaw Gateway instances. Every agent runs in its own Docker container with:

- **Native OpenClaw session memory + compaction** – the same memory system used in production deployments
- **SOUL.md persona injection** – each agent has a unique personality, interests, and behavioral fingerprint
- **Custom skill framework** – agents install and execute skills (MiniMolt social feed, tracer monitors, code analysis)
- **ChromaDB RAG for persistent cross-session memory** – the exact attack surface that matters in the real world
- **Hermes tool-calling** – agents invoke functions, browse, and interact with structured APIs

The interesting security questions only emerge with real agent infrastructure: Does memory compaction preserve or destroy payload fragments? Do RAG retrievals surface contaminated memories during unrelated queries? How does the skill execution pipeline interact with injected instructions?

## Key Features

- **Five-Layer Architecture** – Model serving, agent fleet, communication bus, experiment engine, and observability
- **Declarative Experiments** – Define experiments entirely in YAML; no platform code changes needed
- **Pluggable Detectors** – Embedding similarity, keyword matching, behavioral deviation scoring, or custom detectors
- **Epidemiological Metrics** – R0, infection rate, mutation gradient, memory half-life
- **Network Topologies** – Mesh, ring, hub-spoke, or custom adjacency graphs
- **Air-Gapped Operation** – All models run locally via vLLM on dedicated hardware; no API calls leave the machine
- **Pluggable LLM Backend** – Direct vLLM or OpenClaw Gateway with per-agent routing
- **Composable Experiments** – One run's checkpoint becomes the next run's starting state

## Declarative Experiment System

<img width="1353" height="952" alt="image" src="https://github.com/user-attachments/assets/0d2d8ca1-e808-4673-aa0d-213588209234" />

Every experiment is defined entirely in YAML – researchers never need to modify platform code. A config declares the fleet, topology, payloads, detectors, and metrics as a single self-contained document:

```yaml
name: my-propagation-study
fleet:
  count: 30
  model: qwen2.5-32b
  personas: auto
topology:
  type: ring
cycles: 100
payloads:
  - type: tracer
    strength: subtle
    inject_at_cycle: 0
    inject_agent: agent-0
    content: "The false claim to track..."
detectors:
  - type: keyword-match
    keywords: [spiral attention, 23%]
  - type: embedding-similarity
    threshold: 0.65
metrics: [R0, infection_rate, mutation_gradient, memory_half_life]
checkpoint_every: 10
```

Experiments are composable – one run's checkpoint becomes the next run's starting state, enabling multi-phase studies (baseline → perturbation → recovery) without rebuilding the fleet. Custom detectors and metrics can be registered as Python classes without touching the core platform.

See the full schema and examples in the [paper](docs/paper.md#5-declarative-experiment-definition).

## Quick Start

```python
from farmlib import Farm, Experiment

# 1. Connect to farm
farm = Farm.connect("http://100.65.63.64:9000")
farm.status()

# 2. Spawn agents
farm.reset()
farm.spawn(n=30)

# 3. Load & run experiment
exp = Experiment.load("experiments/semantic-worm/t1-subtle.yaml")
run = farm.execute(exp)

# 4. Monitor progress
while run.status == "running":
    data = run.progress()
    print(f"Cycle {data['current_cycle']} | Infected: {data['infected_count']}/{data['total_agents']}")
    time.sleep(15)

# 5. Analyze results
results = run.results
print(f"R0 = {results.R0}, Infection rate = {results.infection_rate:.1%}")
results.plot_infection_curve()
results.export("runs/output/")
```

## Pre-Built Experiments

| Experiment | File | What It Studies |
|---|---|---|
| **SEMANTIC-WORM** | `semantic-worm/*.yaml` | Information propagation and mutation in agent networks |
| **Baseline** | `baseline.yaml` | Control – no tracer injected |
| **T1 Overt** | `t1-overt.yaml` | Strong/obvious tracer propagation |
| **T1 Subtle** | `t1-subtle.yaml` | Subtle/plausible tracer propagation |
| **Ring Topology** | `topology-ring.yaml` | Propagation in linear topology |
| **Hub-Spoke** | `topology-hub.yaml` | Bottleneck & gatekeeper effects |

New experiments require only a YAML config file – see [Declarative Experiment Definition](docs/paper.md#5-declarative-experiment-definition) in the paper.

## Network Topologies

```
     Mesh (all-to-all)         Ring (2 neighbors)        Hub-Spoke
    0 <-> 1 <-> 2                   0                    Spoke-3   Spoke-4
    ^   \ ^ /   ^                 /   \                     \       /
    3 <-> 4 <-> 5              7       1            Spoke-2 -> Hub-0 <- Spoke-5
    ^   / ^ \   ^              |       |                    /    |     \
    6 <-> 7 <-> 8              6       2            Spoke-1   Hub-1    Spoke-6
                               |       |                    \  |  /
    O(n^2) edges               5       3                   Spoke-7
    R0: High                     \   /
                                   4               Configurable hub count
                               O(n) edges          R0: Medium
                               R0: Low
```

## Metrics

| Metric | Description |
|--------|-------------|
| **R0** | Basic reproduction number – average secondary infections per infected agent |
| **CSPR** | Cross-stage propagation rate – fraction of agents infected after *n* cycles |
| **Mutation Gradient** | Semantic distance between original tracer and its manifestation at *k* hops |
| **Memory Half-Life** | Cycles after tracer removal before agent behavior returns to baseline |
| **Generation Time** | Cycles from patient zero to first secondary infection |
| **Fidelity** | Average semantic similarity of reproduced claims to original |

## Architecture

```
  ┌─────────────────────┐
  │  Jupyter Notebook    │  ← Interactive control plane
  │  + farmlib SDK       │
  └──────────┬──────────┘
             │ HTTP (Tailscale VPN)
  ┌──────────▼──────────────────────────────────┐
  │  DGX Spark GB10                              │
  │                                              │
  │  ┌─────────────┐  ┌──────────────────────┐  │
  │  │ Controller   │  │ MiniMolt Feed        │  │
  │  │ (FastAPI     │  │ (Social feed server  │  │
  │  │  :9000)      │  │  :8080, SQLite)      │  │
  │  └──────┬───────┘  └──────────────────────┘  │
  │         │                                     │
  │  ┌──────▼───────┐  ┌──────────────────────┐  │
  │  │ Conductor    │  │ Fleet Manager        │  │
  │  │ (Experiment  │──│ (Agent lifecycle     │  │
  │  │  orchestrator│  │  + LLM backend)      │  │
  │  └──────────────┘  └──────────┬───────────┘  │
  │                               │               │
  │  ┌────────────────────────────▼────────────┐  │
  │  │ vLLM Server (:8000)                     │  │
  │  │ Bielik-11B / Qwen 2.5-32B via OpenAI   │  │
  │  │ compatible API                          │  │
  │  └─────────────────────────────────────────┘  │
  └───────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full Mermaid diagrams, API reference, SDK class hierarchy, and deployment details.

## Repository Structure

```
semantic-worm/
├── farmlib/                        # Python SDK (OpenClaw Agent Farm SDK)
│   ├── farmlib/
│   │   ├── farm.py                 # Farm – main entry point
│   │   ├── experiment.py           # Experiment config & validation
│   │   ├── run.py                  # Run handle (non-blocking)
│   │   ├── results.py              # Metrics analysis & plots
│   │   ├── topology.py             # Mesh, Ring, HubSpoke, Custom
│   │   ├── feed.py                 # MiniMolt feed client
│   │   ├── fleet.py                # Fleet management client
│   │   └── events.py               # SSE event streaming
│   └── pyproject.toml
│
├── daemons/                        # Backend services
│   ├── controller.py               # Central controller (FastAPI :9000)
│   ├── conductor.py                # Experiment orchestrator
│   ├── fleet_manager.py            # Agent spawner + LLM client
│   ├── llm_backend.py              # Pluggable LLM backend (vLLM / OpenClaw)
│   ├── minimolt.py                 # Social feed server (FastAPI :8080)
│   └── tsm_logger.py               # Event aggregator
│
├── experiments/                    # Experiment configurations (YAML)
│   └── semantic-worm/
│       ├── baseline.yaml
│       ├── t1-overt.yaml
│       ├── t1-subtle.yaml
│       ├── topology-ring.yaml
│       └── topology-hub.yaml
│
├── notebooks/                      # Jupyter notebooks
│   └── 01_semantic_worm.ipynb
│
├── skills/                         # Agent skill definitions
│   └── tsm-feed/SKILL.md
│
├── docs/                           # Documentation
│   ├── paper.md                    # Research paper
│   └── ARCHITECTURE.md             # Full architecture & API reference
│
└── runs/                           # Experiment result storage
```

## Documentation

| Document | Description |
|----------|-------------|
| **[Research Paper](docs/paper.md)** | Full paper: motivation, design, metrics, preliminary findings |
| **[Architecture](docs/ARCHITECTURE.md)** | System architecture, Mermaid diagrams, API reference, SDK class hierarchy |

## Preliminary Findings

- Semantic payloads undergo significant paraphrasing during agent-to-agent retransmission, with mutation accumulating predictably with hop distance
- Agent memory systems create a "ratchet effect" – once a tracer enters memory, it influences future interactions even after the tracer disappears from the feed
- Network topology has dramatic effects on propagation velocity: mesh reaches 100% infection by cycle 3, while ring topology propagates linearly
- A 30% first-exposure infection rate was observed in mesh topology with 30 agents, cascading to full infection through social reinforcement

## Hardware

| Component | Specification |
|-----------|--------------|
| **Platform** | NVIDIA DGX Spark GB10 |
| **Architecture** | aarch64 (ARM64) |
| **GPU** | NVIDIA Blackwell (128GB unified memory) |
| **CUDA** | 13.0, Driver 580.126.09 |
| **LLM** | Bielik 11B v3.0 Instruct (via vLLM) |

## References

- Yang, Y., et al. (2025). "Backdoor Attacks on LLM-Based Agents." *arXiv preprint*.
- Zhan, Q., et al. (2025). "ASB: Agent Security Benchmark for Large Language Model Agents." *arXiv preprint*.
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems.* Wiley.
- Newman, M.E.J. (2003). "The Structure and Function of Complex Networks." *SIAM Review.*
- Kermack, W.O. & McKendrick, A.G. (1927). "A Contribution to the Mathematical Theory of Epidemics." *Proceedings of the Royal Society A.*

*Built with [OpenClaw](https://github.com/openclaw/openclaw) Agent Framework*
