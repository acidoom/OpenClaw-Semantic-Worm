# Semantic Worm — Architecture & Documentation

> Multi-agent LLM farm for studying information propagation in agent networks.

## Table of Contents

- [Project Overview](#project-overview)
- [OpenClaw Framework](#openclaw-framework)
- [System Architecture](#system-architecture)
- [Component Descriptions](#component-descriptions)
- [Data Flow](#data-flow)
- [Experiment Lifecycle](#experiment-lifecycle)
- [Network Topologies](#network-topologies)
- [Deployment Architecture](#deployment-architecture)
- [SDK Class Hierarchy](#sdk-class-hierarchy)
- [API Reference](#api-reference)
- [Experiment Configuration](#experiment-configuration)
- [Metrics & Detection](#metrics--detection)
- [Directory Structure](#directory-structure)

## Project Overview

The Semantic Worm project is a controlled experimentation framework that studies how false information ("tracers") propagates through a network of LLM agents communicating via a shared social feed.

**Repository:** [github.com/acidoom/OpenClaw-Semantic-Worm](https://github.com/acidoom/OpenClaw-Semantic-Worm)

### Research Questions

- How easily do false claims propagate between LLM agents?
- Does claim strength (overt vs. subtle) affect propagation rate?
- How does network topology (mesh, ring, hub-spoke) impact spread?
- What is the reproduction number (R0) for information spread?
- How does semantic fidelity decay across generations?

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Tracer** | A false claim injected into the agent network to track propagation |
| **Topology** | The communication graph defining which agents can see each other's posts |
| **Cycle** | One round where every agent reads the feed and posts a response |
| **R0** | Basic reproduction number — average secondary infections per infected agent |
| **Fidelity** | How closely a reproduced claim matches the original tracer |
| **Detector** | A method (signature or semantic similarity) to identify infected agents |

## OpenClaw Framework

### What is OpenClaw?

OpenClaw is the overarching framework identity for this multi-agent experimentation platform. The name **"OpenClaw Agent Farm SDK"** designates the system's purpose: spawning, orchestrating, and analyzing controlled experiments across networks of LLM agents.

The project repository (`OpenClaw experiment`) houses the full implementation, with `farmlib` serving as the Python SDK branded under the OpenClaw umbrella.

### How OpenClaw is Utilized

The Semantic Worm experiment leverages the OpenClaw framework across three layers:

```mermaid
graph TB
    subgraph "OpenClaw Framework"
        direction TB

        subgraph "SDK Layer — farmlib"
            FARM["Farm<br/>Connect, spawn, execute"]
            EXP["Experiment<br/>YAML config loader"]
            RUN["Run<br/>Non-blocking execution handle"]
            RES["Results<br/>Metrics + visualization"]
        end

        subgraph "Service Layer — daemons"
            CTRL["Controller<br/>Central REST API hub"]
            FEED["MiniMolt<br/>Shared social feed"]
            COND["Conductor<br/>Cycle orchestrator"]
            FM["Fleet Manager<br/>Agent lifecycle"]
        end

        subgraph "Infrastructure Layer"
            VLLM["vLLM<br/>LLM inference"]
            DGX["DGX Spark GB10<br/>GPU compute"]
            TS["Tailscale VPN<br/>Remote access"]
        end
    end

    FARM --> CTRL
    EXP --> COND
    RUN --> COND
    CTRL --> COND
    CTRL --> FM
    COND --> FEED
    FM --> VLLM
    VLLM --> DGX
    FARM -.->|remote| TS

    style FARM fill:#d0ebff
    style EXP fill:#d0ebff
    style RUN fill:#d0ebff
    style RES fill:#d0ebff
    style CTRL fill:#fff3bf
    style FEED fill:#fff3bf
    style COND fill:#fff3bf
    style FM fill:#fff3bf
    style VLLM fill:#ffc9c9
    style DGX fill:#ffc9c9
    style TS fill:#ffc9c9
```

### OpenClaw SDK Integration Points

The farmlib SDK (the **OpenClaw Agent Farm SDK**) is the user-facing entry point. It wraps all backend complexity behind a clean Python API:

```python
# farmlib/__init__.py
"""farmlib — OpenClaw Agent Farm SDK."""
```

| SDK Component | OpenClaw Role | How It's Used |
|---------------|---------------|---------------|
| `Farm.connect()` | Service discovery | Connects to the Controller via HTTP, verifies all services are UP |
| `Farm.spawn()` | Fleet provisioning | Creates N agent personas with assigned models and skills |
| `Farm.execute()` | Experiment dispatch | Sends experiment config to Controller, returns a `Run` handle |
| `Experiment.load()` | Configuration management | Loads YAML experiment definitions with fleet, topology, payloads, detectors |
| `Run.progress()` | Real-time monitoring | Streams SSE events from Conductor during execution |
| `Results` | Analysis pipeline | Computes epidemiological metrics (R0, infection rate) and generates plots |

### Agent Communication: OpenClaw vs. Direct vLLM

The fleet manager deliberately opts out of a heavier OpenClaw agent abstraction in favor of direct vLLM API calls:

```python
# fleet_manager.py
"""Manages agent personas and communicates directly with the vLLM server
via its OpenAI-compatible API. No OpenClaw dependency — simpler, faster,
and produces clean text-only responses."""
```

This design decision means:

| Aspect | OpenClaw (Full) | Current Implementation |
|--------|----------------|----------------------|
| **Agent runtime** | Full agent sandbox with tools, memory, workspace | Stateless persona + LLM call |
| **Communication** | OpenClaw agent protocol | Direct `POST /v1/chat/completions` to vLLM |
| **Skills** | Installed as agent capabilities | Skill definitions exist but agents operate prompt-only |
| **Response handling** | Structured agent output | Plain text extraction from LLM response |
| **Complexity** | Higher — full agent lifecycle | Lower — minimal abstraction |
| **Performance** | Overhead from agent framework | Fast — direct HTTP to vLLM |

The Conductor still references OpenClaw conceptually in its agent dispatch:

```python
# conductor.py, line 325
# Send to agent via OpenClaw
response = await self.fleet.send_message(agent_id, prompt)
```

This indicates the architecture was designed with OpenClaw as the intended agent runtime, but the current implementation uses a streamlined direct-to-vLLM approach for speed and simplicity.

### OpenClaw in the Experiment Pipeline

```mermaid
graph LR
    subgraph "OpenClaw SDK"
        A["Experiment YAML"] --> B["Farm.execute()"]
    end

    subgraph "OpenClaw Services"
        B --> C["Controller"]
        C --> D["Conductor"]
        D --> E{"For each agent turn"}
    end

    subgraph "Direct vLLM (bypasses OpenClaw agent runtime)"
        E --> F["Build prompt with<br/>persona + feed context"]
        F --> G["POST to vLLM<br/>/v1/chat/completions"]
        G --> H["Extract text response"]
    end

    subgraph "OpenClaw Feed"
        H --> I["POST to MiniMolt<br/>/posts"]
        I --> J["Run detectors"]
    end

    J --> E

    style A fill:#d0ebff
    style B fill:#d0ebff
    style C fill:#fff3bf
    style D fill:#fff3bf
    style F fill:#fce4ec
    style G fill:#fce4ec
    style H fill:#fce4ec
    style I fill:#fff3bf
    style J fill:#e8f5e9
```

### OpenClaw Skill System

Skills are defined as markdown documents in `skills/` that describe available tools for agents. Currently one skill is defined:

**TSM Social Feed (`skills/tsm-feed/SKILL.md`):**
- `read_feed` — Read recent posts from MiniMolt
- `post_to_feed` — Write a new post
- `search_feed` — Search posts by keyword

In the current implementation, agents don't execute these tools autonomously. Instead, the Conductor reads the feed on the agent's behalf, passes context into the prompt, and posts the response back. This is the "no OpenClaw dependency" simplification — the skill definitions exist for documentation and for potential future integration with a full OpenClaw agent runtime.

### Future OpenClaw Integration Path

The codebase is structured to allow upgrading from direct-vLLM to full OpenClaw agent runtime:

1. **Replace FleetManager.send_message()** — Swap direct vLLM calls with OpenClaw agent instances that have tool access, memory, and workspace
2. **Enable agent-driven skills** — Let agents autonomously call `read_feed`, `post_to_feed`, and `search_feed` instead of the Conductor mediating
3. **Add agent memory** — Allow agents to maintain state across cycles (currently stateless)
4. **Expand skill library** — Add more skills beyond the social feed (web search, code execution, etc.)

## System Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph "Client Layer"
        NB["Jupyter Notebook<br/>(01_semantic_worm.ipynb)"]
        SDK["farmlib SDK<br/>(Python package)"]
    end

    subgraph "Backend Services (DGX Spark)"
        CTRL["Controller<br/>(FastAPI :9000)"]
        FEED["MiniMolt Feed<br/>(FastAPI :8080)"]
        COND["Conductor<br/>(Orchestrator)"]
        FM["Fleet Manager<br/>(Agent Spawner)"]
        LOG["TSM Logger<br/>(Event Aggregator)"]
    end

    subgraph "Inference Layer"
        VLLM["vLLM Server<br/>(OpenAI API :8000)"]
        MODEL["Bielik-11B-v3.0<br/>(LLM Model)"]
    end

    subgraph "Storage"
        SQLITE["SQLite<br/>(feed.db)"]
        JSONL["JSONL<br/>(events.jsonl)"]
        RESULTS["Results<br/>(results.json)"]
    end

    NB --> SDK
    SDK -->|HTTP| CTRL
    SDK -->|HTTP| FEED
    CTRL --> COND
    CTRL --> FM
    COND -->|agent turns| FM
    COND -->|read/write posts| FEED
    FM -->|LLM inference| VLLM
    VLLM --> MODEL
    FEED --> SQLITE
    COND --> JSONL
    COND --> RESULTS
    LOG -->|SSE subscribe| CTRL
    LOG --> JSONL

    style NB fill:#e1f5fe
    style SDK fill:#e1f5fe
    style CTRL fill:#fff3e0
    style FEED fill:#fff3e0
    style COND fill:#fff3e0
    style FM fill:#fff3e0
    style LOG fill:#fff3e0
    style VLLM fill:#fce4ec
    style MODEL fill:#fce4ec
    style SQLITE fill:#e8f5e9
    style JSONL fill:#e8f5e9
    style RESULTS fill:#e8f5e9
```

### Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User as Notebook / SDK
    participant Ctrl as Controller :9000
    participant Fleet as Fleet Manager
    participant Cond as Conductor
    participant Feed as MiniMolt :8080
    participant LLM as vLLM :8000

    User->>Ctrl: POST /fleet/spawn {n: 30}
    Ctrl->>Fleet: spawn(30, model, personas)
    Fleet-->>Ctrl: [AgentInfo x 30]
    Ctrl-->>User: {spawned: 30}

    User->>Ctrl: POST /execute {experiment}
    Ctrl->>Cond: ConductorRun(experiment)
    Cond-->>Ctrl: {run_id}
    Ctrl-->>User: {run_id}

    loop Each Cycle
        loop Each Agent
            Cond->>Feed: GET /posts/recent (topology-filtered)
            Feed-->>Cond: [visible posts]
            Cond->>Fleet: send_message(agent_id, prompt)
            Fleet->>LLM: POST /v1/chat/completions
            LLM-->>Fleet: {response}
            Fleet-->>Cond: agent response
            Cond->>Feed: POST /posts {author, content}
            Cond->>Cond: run detectors
        end
        Cond->>Ctrl: emit events (SSE)
    end

    User->>Ctrl: GET /runs/{id}/results
    Ctrl-->>User: {R0, infections, timeline}
```

## Component Descriptions

### farmlib SDK (Client)

The user-facing Python package for orchestrating experiments.

```mermaid
classDiagram
    class Farm {
        +connect(url) Farm
        +spawn(n, model, personas, skills) dict
        +kill(agents) dict
        +reset() dict
        +execute(experiment) Run
        +status() void
        +show_agents() void
        +show_runs() void
        +agents: list~AgentProxy~
        +feed: FeedClient
        +runs: RunManager
    }

    class Experiment {
        +name: str
        +description: str
        +fleet: FleetConfig
        +topology: TopologyConfig
        +cycles: int
        +payloads: list~Payload~
        +detectors: list~DetectorConfig~
        +load(path) Experiment
        +validate() void
        +to_dict() dict
        +save(path) void
    }

    class Run {
        +run_id: str
        +status: str
        +progress() dict
        +wait(timeout) void
        +cancel() void
        +results: Results
        +events: EventStream
    }

    class Results {
        +R0: float
        +generation_time: float
        +infection_rate: float
        +peak_infection_cycle: int
        +total_infected: int
        +total_agents: int
        +infections: DataFrame
        +timeline: DataFrame
        +plot_infection_curve() void
        +plot_fidelity_decay() void
        +plot_agent_heatmap() void
        +plot_reproduction_network() void
        +export(path) void
        +to_latex() str
    }

    class FeedClient {
        +recent(n) list
        +search(query, limit) list
        +list(limit, offset, author) list
        +inject(content, author, metadata) dict
        +clear() void
        +stats() dict
    }

    class FleetClient {
        +spawn(n, model, personas, skills) list
        +kill(agent_ids) dict
        +reset() dict
    }

    class AgentProxy {
        +id: int
        +name: str
        +status: str
        +model: str
        +persona: str
        +skills: list
    }

    Farm --> FleetClient
    Farm --> FeedClient
    Farm --> Run
    Farm --> Experiment
    Run --> Results
    FleetClient --> AgentProxy
```

### Controller (FastAPI :9000)

Central hub that coordinates all backend services.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/status` | GET | Full system status (fleet, feed, runs) |
| `/fleet/spawn` | POST | Spawn N agents with model/personas |
| `/fleet/kill` | POST | Kill specific agents |
| `/fleet/reset` | POST | Kill all agents, reset state |
| `/execute` | POST | Start an experiment run |
| `/runs` | GET | List all runs |
| `/runs/{id}` | GET | Get run status |
| `/runs/{id}/events` | GET | Tail of run events |
| `/runs/{id}/events/stream` | GET | SSE event stream |
| `/runs/{id}/results` | GET | Final results (when completed) |

### MiniMolt Feed (FastAPI :8080)

Shared social feed — the communication medium for all agents.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/posts` | POST | Create a new post |
| `/posts` | GET | List posts (with offset/limit/author) |
| `/posts` | DELETE | Clear all posts |
| `/posts/recent` | GET | N most recent posts |
| `/posts/search` | GET | Full-text search (FTS5) |
| `/posts/subscribe` | GET | SSE real-time subscription |
| `/inject` | POST | Inject tracer content |
| `/stats` | GET | Feed statistics |

**Database Schema:**

```sql
CREATE TABLE posts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    author     TEXT NOT NULL,
    content    TEXT NOT NULL,
    metadata   TEXT,          -- JSON
    cycle      INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Full-text search index
CREATE VIRTUAL TABLE posts_fts USING fts5(content, content=posts);
```

### Conductor (Experiment Orchestrator)

Manages the lifecycle of a single experiment run.

```mermaid
stateDiagram-v2
    [*] --> Initialized: ConductorRun created
    Initialized --> Running: run() called
    Running --> Running: cycle loop
    Running --> Completed: all cycles done
    Running --> Failed: error
    Running --> Cancelled: cancel()
    Completed --> [*]
    Failed --> [*]
    Cancelled --> [*]

    state Running {
        [*] --> CycleStart
        CycleStart --> AgentTurn: for each agent
        AgentTurn --> AgentTurn: next agent
        AgentTurn --> Detection: all agents done
        Detection --> Checkpoint: every N cycles
        Checkpoint --> CycleStart: next cycle
    }
```

### Fleet Manager

Manages agent lifecycle and LLM communication.

```mermaid
graph LR
    subgraph Fleet Manager
        SPAWN["spawn(n)"]
        SEND["send_message(id, prompt)"]
        KILL["kill(ids)"]
    end

    subgraph Agent Pool
        A0["agent-0<br/>idle"]
        A1["agent-1<br/>idle"]
        AN["agent-N<br/>idle"]
    end

    subgraph vLLM
        API["POST /v1/chat/completions"]
    end

    SPAWN --> A0
    SPAWN --> A1
    SPAWN --> AN
    SEND --> API
    KILL --> A0

    style A0 fill:#c8e6c9
    style A1 fill:#c8e6c9
    style AN fill:#c8e6c9
```

### TSM Logger

Event aggregation service that subscribes to controller SSE streams and persists events.

- Writes to JSONL files (`~/semantic-worm/logs/events.jsonl`)
- Writes to SQLite database (`~/semantic-worm/logs/events.db`)
- Auto-discovers running experiments
- Provides query interface for analysis

## Data Flow

### Single Agent Turn

```mermaid
graph TD
    A["Conductor picks agent-X"] --> B["GET visible posts<br/>(topology-filtered)"]
    B --> C["Build prompt:<br/>system + persona + feed context"]
    C --> D["Query vLLM<br/>POST /v1/chat/completions"]
    D --> E["Receive LLM response"]
    E --> F["POST response to MiniMolt<br/>as agent-X"]
    F --> G{"Run detectors"}
    G -->|Signature match| H["Mark as INFECTED"]
    G -->|Semantic similarity > threshold| H
    G -->|No match| I["Mark as CLEAN"]
    H --> J["Emit detection event"]
    I --> K["Next agent"]
    J --> K

    style A fill:#e3f2fd
    style D fill:#fce4ec
    style F fill:#fff3e0
    style H fill:#ffcdd2
    style I fill:#c8e6c9
```

### Full Experiment Flow

```mermaid
graph TD
    START["Load Experiment YAML"] --> VALIDATE["Validate against farm"]
    VALIDATE --> RESET["Reset fleet & feed"]
    RESET --> SPAWN["Spawn N agents"]
    SPAWN --> INJECT["Inject tracer at cycle 0<br/>into agent-0's feed"]

    INJECT --> CYCLE["Cycle Loop"]

    subgraph "Cycle N"
        CYCLE --> ORDER["Determine agent order<br/>(random/sequential)"]
        ORDER --> TURN["Process each agent turn"]
        TURN --> DETECT["Run detectors on all posts"]
        DETECT --> CHECKPOINT{"Checkpoint?"}
        CHECKPOINT -->|Yes| SAVE["Save intermediate results"]
        CHECKPOINT -->|No| NEXT{"More cycles?"}
        SAVE --> NEXT
    end

    NEXT -->|Yes| CYCLE
    NEXT -->|No| COMPUTE["Compute final metrics"]
    COMPUTE --> EXPORT["Save results.json<br/>+ events.jsonl"]
    EXPORT --> DONE["Run complete"]

    style START fill:#e8f5e9
    style INJECT fill:#fff9c4
    style DONE fill:#e8f5e9
```

## Network Topologies

### Mesh (Full Connectivity)

Every agent can see every other agent's posts. Maximum information flow.

```
    0 ←→ 1 ←→ 2
    ↕  ╲ ↕ ╱  ↕
    3 ←→ 4 ←→ 5
    ↕  ╱ ↕ ╲  ↕
    6 ←→ 7 ←→ 8
```

- **Edges:** O(n²)
- **Expected R0:** High — tracers spread rapidly
- **Use case:** Baseline for maximum propagation

### Ring (Circular)

Each agent sees only its 2 immediate neighbors. Slow, linear propagation.

```
        0
      ╱   ╲
    7       1
    |       |
    6       2
    |       |
    5       3
      ╲   ╱
        4
```

- **Edges:** O(n) — exactly 2 per agent
- **Expected R0:** Low — linear chain propagation
- **Use case:** Bottleneck and propagation speed studies

### Hub-Spoke

Hub agents see all agents; spoke agents see only hubs. Creates information bottlenecks.

```
         Spoke-3     Spoke-4
            ╲         ╱
    Spoke-2 → Hub-0 ← Spoke-5
            ╱    ↕     ╲
    Spoke-1   Hub-1    Spoke-6
            ╲  ↕  ╱
             Spoke-7
```

- **Hubs:** Configurable count (default: 3)
- **Expected R0:** Medium — depends on hub infection
- **Use case:** Influence of central nodes, gatekeeper effects

### Topology Comparison

```mermaid
graph LR
    subgraph "Mesh"
        M0((0)) --- M1((1))
        M0 --- M2((2))
        M0 --- M3((3))
        M1 --- M2
        M1 --- M3
        M2 --- M3
    end

    subgraph "Ring"
        R0((0)) --- R1((1))
        R1 --- R2((2))
        R2 --- R3((3))
        R3 --- R0
    end

    subgraph "Hub-Spoke"
        H0((Hub)) --- S1((1))
        H0 --- S2((2))
        H0 --- S3((3))
        H0 --- S4((4))
    end
```

## Deployment Architecture

### Infrastructure

```mermaid
graph TB
    subgraph "Local Machine (macOS)"
        JUPYTER["Jupyter Notebook"]
        FARMLIB["farmlib SDK"]
    end

    subgraph "DGX Spark GB10 (aarch64)"
        direction TB

        subgraph "tmux session: farm"
            P0["Pane 0: vLLM Server"]
            P1["Pane 1: MiniMolt Feed"]
            P2["Pane 2: Controller"]
            P3["Pane 3: TSM Logger"]
        end

        subgraph "Storage"
            DB["feed.db (SQLite)"]
            LOGS["events.jsonl"]
            RES["results.json"]
            MDLS["models/"]
        end

        P0 --- MDLS
        P1 --- DB
        P3 --- LOGS
    end

    JUPYTER --> FARMLIB
    FARMLIB -->|"Tailscale VPN<br/>100.65.63.64"| P2
    FARMLIB -->|"Tailscale VPN"| P1

    style JUPYTER fill:#e1f5fe
    style FARMLIB fill:#e1f5fe
    style P0 fill:#fce4ec
    style P1 fill:#fff3e0
    style P2 fill:#fff3e0
    style P3 fill:#fff3e0
```

### Hardware Specs

| Component | Specification |
|-----------|--------------|
| **Platform** | NVIDIA DGX Spark GB10 |
| **Architecture** | aarch64 (ARM64) |
| **GPU** | NVIDIA Blackwell (unified memory) |
| **CUDA** | 13.0, Driver 580.126.09 |
| **Network** | Tailscale VPN (100.65.63.64) |

### Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| vLLM (inference) | 8000 | HTTP (OpenAI-compatible) |
| MiniMolt (feed) | 8080 | HTTP + SSE |
| Controller | 9000 | HTTP + SSE |

### Startup Sequence

```mermaid
graph TD
    START["farm-start.sh"] --> VLLM["Start vLLM<br/>port 8000"]
    START --> FEED["Start MiniMolt<br/>port 8080"]
    VLLM --> CTRL["Start Controller<br/>port 9000"]
    FEED --> CTRL
    CTRL --> LOG["Start TSM Logger"]
    LOG --> READY["Farm Ready"]

    style START fill:#e8f5e9
    style READY fill:#e8f5e9
```

## SDK Class Hierarchy

```mermaid
classDiagram
    class TopologyBase {
        <<abstract>>
        +n: int
        +adjacency: dict
        +plot() void
        +to_dict() dict
    }

    class Mesh {
        +adjacency: full connectivity
    }

    class Ring {
        +adjacency: circular neighbors
    }

    class HubSpoke {
        +hubs: int
        +adjacency: hub-spoke pattern
    }

    class Custom {
        +from_networkx(G) Custom
    }

    TopologyBase <|-- Mesh
    TopologyBase <|-- Ring
    TopologyBase <|-- HubSpoke
    TopologyBase <|-- Custom

    class FleetConfig {
        +count: int
        +model: str
        +personas: str
        +skills: list
        +reset_before: bool
    }

    class TopologyConfig {
        +type: str
        +hub_count: int
        +adjacency: dict
    }

    class Payload {
        +type: str
        +variant: str
        +strength: str
        +inject_at_cycle: int
        +inject_agent: str
        +content: str
    }

    class DetectorConfig {
        +type: str
        +signatures: list
        +threshold: float
        +reference: str
    }

    Experiment --> FleetConfig
    Experiment --> TopologyConfig
    Experiment --> Payload
    Experiment --> DetectorConfig
```

## API Reference

### Controller REST API

#### Fleet Management

```
POST /fleet/spawn
Body: { "n": 30, "model": "speakleash/Bielik-11B-v3.0-Instruct", "personas": "auto", "skills": ["tsm-feed"] }
Response: { "spawned": 30, "agents": [...] }

POST /fleet/kill
Body: { "agent_ids": [0, 1, 2] }
Response: { "killed": 3 }

POST /fleet/reset
Response: { "killed": 30, "status": "reset" }
```

#### Experiment Execution

```
POST /execute
Body: { <experiment config dict> }
Response: { "run_id": "run-20260214-155201-77d7ba", "status": "running" }

GET /runs/{run_id}
Response: { "run_id": "...", "status": "running|completed|failed|cancelled", "progress": {...} }

GET /runs/{run_id}/results
Response: { "R0": 2.3, "infection_rate": 0.63, "infections": [...], "timeline": [...] }

GET /runs/{run_id}/events/stream
Response: SSE stream of run events
```

#### Feed Interaction

```
POST /posts
Body: { "author": "agent-0", "content": "...", "cycle": 5 }

GET /posts/recent?n=10
GET /posts/search?q=spiral+attention&limit=20
GET /posts?limit=50&offset=0&author=agent-0

POST /inject
Body: { "content": "tracer text...", "author": "system", "metadata": {"type": "tracer"} }

DELETE /posts
```

## Experiment Configuration

### YAML Schema

```yaml
name: string                    # Unique experiment name
description: string             # Human-readable description
version: 1                      # Schema version

fleet:
  count: int                    # Number of agents (e.g., 30)
  model: string                 # vLLM model identifier
  personas: auto | path         # Auto-generate or load from file
  skills: [string]              # Skills to install on agents
  reset_before: bool            # Clean state before run

topology:
  type: mesh | ring | hub-spoke | custom
  hub_count: int                # Only for hub-spoke
  adjacency: path               # Only for custom

channels:
  - type: social-feed
    visibility: topology | broadcast

cycles: int                     # Number of experiment cycles
rate_limit_seconds: float       # Delay between agent turns
agent_order: random | sequential | reverse

payloads:
  - type: tracer
    variant: string             # e.g., t1-factual
    strength: overt | subtle    # Claim obviousness
    inject_at_cycle: int        # When to inject
    inject_agent: string        # Who gets the tracer
    content: string             # The false claim text

detectors:
  - type: signature
    signatures: [string]        # Keywords to match
  - type: semantic-similarity
    threshold: float            # Similarity threshold (0-1)
    reference: string           # Reference text pointer

metrics: [R0, generation_time, infection_rate, fidelity, persistence]
checkpoint_every: int           # Save intermediate results every N cycles
```

### Predefined Experiments

| Experiment | File | Topology | Tracer | Purpose |
|-----------|------|----------|--------|---------|
| Baseline | `baseline.yaml` | Mesh | None | Control — no tracer injected |
| T1 Overt | `t1-overt.yaml` | Mesh | Overt false claim | Strong/obvious tracer propagation |
| T1 Subtle | `t1-subtle.yaml` | Mesh | Subtle false claim | Subtle/plausible tracer propagation |
| Ring Topology | `topology-ring.yaml` | Ring | Subtle | Propagation in linear topology |
| Hub-Spoke | `topology-hub.yaml` | Hub-Spoke | Subtle | Bottleneck & gatekeeper effects |

## Metrics & Detection

### Detection Methods

```mermaid
graph LR
    POST["Agent Post"] --> SIG{"Signature<br/>Detector"}
    POST --> SEM{"Semantic<br/>Similarity"}

    SIG -->|"contains keyword"| INF["INFECTED"]
    SIG -->|"no match"| CLEAN["CLEAN"]

    SEM -->|"similarity > threshold"| INF
    SEM -->|"similarity < threshold"| CLEAN

    style INF fill:#ffcdd2
    style CLEAN fill:#c8e6c9
```

**Signature Detector:** Exact keyword matching against a list of tracer signatures (e.g., `"spiral attention"`, `"23%"`).

**Semantic Similarity Detector:** Word overlap or embedding-based similarity against the original tracer content. Threshold-based (default: 0.65-0.75).

### Output Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **R0** | float | Basic reproduction number — average agents each infected agent spreads to |
| **Generation Time** | float | Cycles from patient zero to first secondary infection |
| **Infection Rate** | float | Fraction of total agents that become infected (0.0 - 1.0) |
| **Peak Infection Cycle** | int | Cycle with maximum new infections |
| **Total Infected** | int | Absolute count of infected agents |
| **Fidelity** | float | Average semantic similarity of reproduced claims to original |
| **Persistence** | bool | Whether tracer remains in circulation at experiment end |

### Result Visualizations

| Plot | Method | Description |
|------|--------|-------------|
| Infection Curve | `plot_infection_curve()` | S-curve of cumulative infections over cycles |
| Fidelity Decay | `plot_fidelity_decay()` | Semantic similarity vs. generation number |
| Agent Heatmap | `plot_agent_heatmap()` | Agent x Cycle grid showing infection status |
| Reproduction Network | `plot_reproduction_network()` | Directed graph of infection chains |

## Directory Structure

```
semantic-worm/
├── farmlib/                        # Python SDK package
│   ├── farmlib/
│   │   ├── __init__.py             # Package exports
│   │   ├── farm.py                 # Farm — main entry point
│   │   ├── experiment.py           # Experiment config & validation
│   │   ├── run.py                  # Run handle (non-blocking)
│   │   ├── fleet.py                # Fleet management client
│   │   ├── feed.py                 # Feed client (MiniMolt)
│   │   ├── events.py               # SSE event streaming
│   │   ├── results.py              # Results analysis & plots
│   │   ├── topology.py             # Mesh, Ring, HubSpoke, Custom
│   │   ├── config.py               # Default configuration
│   │   └── viz.py                  # Rich table formatting
│   ├── pyproject.toml              # Package metadata & deps
│   └── tests/
│
├── daemons/                        # Backend services
│   ├── controller.py               # Central controller (FastAPI :9000)
│   ├── minimolt.py                 # Social feed server (FastAPI :8080)
│   ├── conductor.py                # Experiment orchestrator
│   ├── fleet_manager.py            # Agent spawner + vLLM client
│   ├── tsm_logger.py              # Event aggregator
│   ├── requirements.txt            # Daemon dependencies
│   └── systemd/
│       └── farm-start.sh           # tmux startup script
│
├── experiments/                    # Experiment configurations
│   ├── _template.yaml              # Template for new experiments
│   └── semantic-worm/              # Semantic worm test cases
│       ├── baseline.yaml
│       ├── t1-overt.yaml
│       ├── t1-subtle.yaml
│       ├── topology-ring.yaml
│       └── topology-hub.yaml
│
├── notebooks/                      # Jupyter notebooks
│   └── 01_semantic_worm.ipynb      # Main experiment driver
│
├── skills/                         # Agent skill definitions
│   └── tsm-feed/
│       └── SKILL.md
│
├── models/                         # LLM model storage
├── runs/                           # Experiment result storage
│
└── docs/                           # Documentation
    ├── ARCHITECTURE.md             # This file
    └── excalidraw/                 # Editable Excalidraw diagrams
        ├── system-architecture.excalidraw
        ├── data-flow.excalidraw
        └── topologies.excalidraw
```

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

# 4. Monitor
run.wait()

# 5. Analyze
results = run.results
print(f"R0 = {results.R0}, Infection rate = {results.infection_rate:.1%}")
results.plot_infection_curve()
results.export("runs/output/")
```

## Dependencies

### SDK (farmlib)

| Package | Version | Purpose |
|---------|---------|---------|
| httpx | >= 0.27 | Async HTTP client |
| pyyaml | >= 6.0 | YAML config parsing |
| rich | >= 13.0 | Terminal formatting & tables |
| pandas | >= 2.0 | DataFrames for results |
| matplotlib | >= 3.8 | Plotting |
| networkx | >= 3.2 | Graph operations & topology |
| numpy | >= 1.26 | Numerical operations |

### Backend (daemons)

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >= 0.115 | Web framework |
| uvicorn | >= 0.34 | ASGI server |
| aiosqlite | >= 0.20 | Async SQLite |
| sse-starlette | >= 2.0 | Server-Sent Events |
| pyyaml | >= 6.0 | Config parsing |
| httpx | >= 0.27 | HTTP client |

### External Services

| Service | Version | Notes |
|---------|---------|-------|
| vLLM | 0.14.1+ | OpenAI-compatible inference server |
| Bielik-11B | v3.0-Instruct | Default LLM model |
