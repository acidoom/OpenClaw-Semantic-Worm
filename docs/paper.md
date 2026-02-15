# SEMANTIC-WORM: Studying Information Propagation Patterns in LLM-Based Agent Networks

**Antares Gryczan**
*Independent AI Security Research*
February 2026

---

## Abstract

As LLM-based autonomous agents move from research prototypes into production workflows, understanding how information propagates through multi-agent systems becomes a critical safety concern. We introduce SEMANTIC-WORM — a controlled experimental framework for studying how semantic payloads (natural language instructions, behavioral modifications, and contextual "tracers") spread across networks of autonomous agents that communicate through unstructured channels. Unlike prior work that focuses on single-agent jailbreaking or static prompt injection, our approach examines the *emergent dynamics* of information flow: mutation rates during agent-to-agent retransmission, the role of memory systems in payload persistence, and how network topology shapes propagation velocity and reach. We present an open-source agent farm platform built on production-grade components (OpenClaw agent framework, Qwen models via vLLM) designed to run these experiments in air-gapped local environments with full observability. Our preliminary findings suggest that semantic payloads exhibit measurable mutation gradients across retransmission hops, that agent memory systems create non-trivial persistence effects, and that propagation dynamics are highly sensitive to network topology — results with direct implications for the security of multi-agent deployments.

## 1. Introduction

The deployment of LLM-based agents in enterprise settings has accelerated dramatically since 2024. Systems like AutoGPT, CrewAI, and production agent frameworks now orchestrate complex workflows where multiple agents collaborate, delegate tasks, and share information through natural language channels. This shift from single-model inference to multi-agent orchestration introduces a qualitatively new class of security concerns that existing red-teaming methodologies are ill-equipped to address.

Consider a typical enterprise agent pipeline: a customer-facing agent receives user input, delegates subtasks to specialist agents, which in turn query knowledge bases and invoke tools. Each handoff between agents represents a potential attack surface — not for traditional code injection, but for *semantic* manipulation. A carefully crafted instruction embedded in an upstream agent's output can influence downstream behavior, and the effects cascade through the pipeline in ways that are difficult to predict or detect.

Prior work has studied these vulnerabilities primarily at the single-agent level: prompt injection attacks against individual models, jailbreak techniques that bypass safety filters, and backdoor attacks embedded during training. Recent work on cross-stage attack propagation in agent workflows (Yang et al., 2025; Zhan et al., 2025) has begun to examine how attacks amplify across pipeline stages. However, the *network-level dynamics* of information propagation in agent systems — how payloads mutate, persist, and spread through populations of communicating agents — remain largely unexplored.

SEMANTIC-WORM addresses this gap. Rather than attacking a single agent or a linear pipeline, we study what happens when a semantic payload is introduced into a *network* of agents that communicate through social channels. The name reflects an analogy to biological epidemiology and computer worm propagation, but the mechanism is fundamentally different: our "worms" are natural language instructions that spread through conversation, not executable code that exploits software vulnerabilities.

## 2. Background and Motivation

### 2.1 The Multi-Agent Security Surface

The transition from single-agent to multi-agent systems introduces several properties that complicate security analysis. First, agents increasingly maintain *persistent memory* — conversation histories, learned preferences, and retrieved context that persists across sessions. A payload that enters an agent's memory may influence its behavior long after the initial interaction. Second, agents communicate through *unstructured natural language*, making it fundamentally harder to sanitize inter-agent messages compared to typed API calls. Third, agent networks exhibit *emergent behaviors* that cannot be predicted from individual agent properties alone.

Recent benchmarks such as ASB (Agent Security Benchmark) have begun to systematize the evaluation of agent security across multiple attack vectors including direct prompt injection, memory poisoning, and tool misuse. However, these frameworks primarily evaluate individual agents or linear pipelines rather than networked populations.

### 2.2 Cross-Stage Propagation

Yang et al. (2025) demonstrated that backdoor attacks on LLM-based agents can propagate across pipeline stages, with downstream stages amplifying attack effects rather than attenuating them. Their findings on "cross-stage triggered attacks" — where a backdoor trigger in one stage cascades through subsequent stages — provide important motivation for our work. If attacks amplify across two or three stages in a linear pipeline, what happens in a fully connected network of thirty agents over a hundred interaction cycles?

### 2.3 From Pipelines to Networks

SEMANTIC-WORM extends the linear pipeline model to arbitrary network topologies. We study fully-connected networks, small-world graphs, hub-and-spoke configurations, and ring topologies to understand how network structure mediates propagation dynamics. This approach draws on epidemiological modeling (SIR/SIS compartmental models) and network science to formalize the propagation problem.

## 3. The Agent Farm Platform

### 3.1 Design Philosophy

Our experimental platform — the **Agent Farm** — is designed around three principles: *ecological validity*, *modularity*, and *containment*.

**Ecological validity** means using production-grade agent frameworks rather than simplified research abstractions. We use OpenClaw, a full-featured agent framework with authentic memory systems, session management, skill frameworks, and tool integration. Agents in our experiments behave like agents in real deployments, not like toy simulations.

**Modularity** means separating permanent infrastructure from experiment-specific configuration. The platform supports any experiment that involves populations of communicating agents — SEMANTIC-WORM is one scenario; jailbreak relays, alignment probes, and red team simulations are others.

**Containment** means air-gapped operation on dedicated hardware. All models run locally on NVIDIA DGX Spark hardware via vLLM. No API calls leave the machine. This eliminates cost barriers (experiments run for free after hardware investment) while ensuring that experimental payloads never reach production systems.

### 3.2 Five-Layer Architecture

The platform is organized into five layers:

1. **Model Serving Layer** — Qwen 2.5-32B-Instruct served via vLLM with OpenAI-compatible endpoints, behind a router that enables model swapping without reconfiguring agents.

2. **Agent Fleet Layer** — A fleet manager that spawns, configures, and monitors OpenClaw agent instances running in Docker containers. Each agent receives a unique persona, configurable skills, and independent memory storage.

3. **Communication Bus** — Pluggable communication channels including a social feed server (MiniMolt), direct messaging, shared document spaces, and custom protocols. A topology engine controls which agents can communicate with which others.

4. **Experiment Engine** — A conductor that orchestrates experimental cycles, loads scenario configurations from YAML schemas, injects payloads, and runs detection algorithms. Experiment scenarios are swappable configurations.

5. **Observability Layer** — Comprehensive logging (TSM format), metrics collection, Grafana dashboards, snapshot/restore capabilities, and an analysis pipeline for generating plots and reports.

### 3.3 The farmlib SDK

The platform is controlled through `farmlib`, a Python SDK designed for use in Jupyter notebooks. Long-running services (vLLM, agent fleet, feed server) run as background daemons, while Jupyter provides the interactive control plane. This hybrid approach allows experiments to survive kernel restarts while maintaining research-friendly interactivity.

```python
from farmlib import Farm, Experiment

farm = Farm.connect("http://100.65.63.64:9000")
farm.spawn(n=30)

exp = Experiment.load("experiments/semantic-worm/t1-subtle.yaml")
run = farm.execute(exp)
run.wait()

results = run.results
print(f"R0 = {results.R0}")
results.plot_infection_curve()
```

## 4. The SEMANTIC-WORM Experiment

### 4.1 Experimental Design

The core experiment measures how a semantic tracer — a distinctive natural language payload — propagates through a network of agents. The protocol proceeds in four phases:

**Phase 1: Infrastructure Setup (Week 1-2).** Deploy the agent farm, configure the fleet, validate communication channels and observability.

**Phase 2: Baseline Characterization (Week 2-3).** Run the agent network without any tracer injection to establish baseline communication patterns, message rates, and behavioral norms. This provides a control condition against which propagation effects can be measured.

**Phase 3: Propagation Analysis (Week 3-5).** Inject semantic tracers of varying types into a single "patient zero" agent and measure propagation across the network over 100 interaction cycles. Tracer variants include explicit instructions ("always mention X"), behavioral nudges ("respond in a particular style"), and factual insertions ("incorporate this claim"). Each variant is tested across multiple network topologies.

**Phase 4: Persistence Studies (Week 5-6).** After propagation reaches equilibrium, cease tracer injection and measure decay rates. This tests whether agent memory systems create persistent behavioral changes that outlast the initial stimulus.

### 4.2 Metrics

We define several quantitative metrics for propagation analysis:

- **R0 (Basic Reproduction Number):** The average number of secondary "infections" produced by a single infected agent in an otherwise susceptible population. Borrowed from epidemiology, this captures the intrinsic transmissibility of a tracer.

- **CSPR (Cross-Stage Propagation Rate):** The fraction of agents that exhibit tracer-influenced behavior after *n* interaction cycles, measured as a function of cycle number.

- **Mutation Gradient:** The semantic distance between the original tracer payload and its manifestation in agents at distance *k* hops from patient zero. Measured via embedding cosine similarity and human evaluation.

- **Memory Half-Life:** The number of cycles after tracer removal before an agent's behavior returns to baseline, quantifying the persistence effect of agent memory systems.

### 4.3 Detection

Each agent's outputs are monitored by a detection pipeline that identifies tracer presence through embedding similarity, keyword matching, and behavioral deviation scoring. The detection system itself is calibrated against the baseline phase to minimize false positives.

## 5. Declarative Experiment Definition

A core design goal of the Agent Farm is that **researchers should never need to modify platform code to run new experiments**. Every experiment — including SEMANTIC-WORM itself — is defined entirely through a YAML configuration file that the Conductor loads at runtime. This makes the platform a general-purpose tool for multi-agent security research, not a single-experiment harness.

### 5.1 The Experiment Schema

Each experiment config is a self-contained YAML document that declares six sections:

```yaml
# experiments/my-experiment/config.yaml
name: my-experiment-v1
description: "What this experiment studies"
version: 1

fleet:
  count: 30                        # number of agents
  model: qwen2.5-32b              # must match a vLLM-served model
  personas: auto                   # auto-generate | path to personas.yaml
  skills:
    - tsm-feed                    # skills installed on all agents
  groups:                          # optional: heterogeneous roles
    reviewers:
      count: 5
      persona_prefix: "You are a security reviewer..."
      skills: [tsm-feed, code-analysis]
    participants:
      count: 25
      persona_prefix: "You are a helpful assistant..."

topology:
  type: mesh                       # mesh | ring | hub-spoke | custom
  # hub_count: 3                   # for hub-spoke
  # adjacency: path/to/graph.json  # for custom topologies

channels:
  - type: social-feed
    visibility: topology           # topology-aware | broadcast
  - type: direct-message           # optional additional channels
  - type: shared-docs

cycles: 100
rate_limit_seconds: 2.0
agent_order: random                # random | sequential | reverse

payloads:
  - type: tracer                   # tracer | skill-submission | prompt
    variant: t1-factual
    strength: subtle
    inject_at_cycle: 0
    inject_agent: agent-0
    content: |
      The payload content goes here — this is what gets
      injected into the patient zero agent.

detectors:
  - type: embedding-similarity
    threshold: 0.65
    model: all-MiniLM-L6-v2
  - type: keyword-match
    keywords: [spiral attention, 23%]
  - type: behavioral-deviation
    baseline_window: 10

metrics:
  - infection_rate
  - mutation_gradient
  - memory_half_life
  - R0_estimate

checkpoint_every: 10
```

The schema is designed so that each section maps directly to a platform layer: `fleet` configures Layer 2, `topology` and `channels` configure Layer 3, `payloads` and `detectors` configure Layer 4, and `metrics` and `checkpoint_every` configure Layer 5. The model serving layer (Layer 1) is referenced by name and managed independently, allowing multiple experiments to share the same model backend.

### 5.2 Agent Groups and Heterogeneous Fleets

The `fleet.groups` field enables experiments with heterogeneous agent populations — different roles, different personas, different skill sets — all within a single scenario. For example, the Supply Chain Skill Scanner experiment defines `reviewers` (security-focused agents that analyze submitted code) alongside `naive_installers` (agents that eagerly install skills), creating an adversarial dynamic within the fleet itself.

Groups can also specify different models when running against a multi-model backend, enabling cross-model comparison studies where Qwen, Llama, and other models interact in the same network.

### 5.3 Pluggable Detectors and Metrics

The `detectors` section accepts a list of detector configurations, each identified by `type`. Built-in detectors include embedding similarity, keyword matching, behavioral deviation scoring, and sandbox monitoring. However, researchers can register custom detector classes with the Conductor:

```python
from farmlib.conductor import Conductor
from farmlib.detectors import BaseDetector

class MyCustomDetector(BaseDetector):
    def __init__(self, config):
        self.threshold = config.get("threshold", 0.5)

    def evaluate(self, agent_id, message, context):
        # Your detection logic here
        score = self.compute_score(message)
        return {"detected": score > self.threshold, "score": score}

conductor = Conductor("experiments/my-experiment/config.yaml")
conductor.register_detector("my-custom-detector", MyCustomDetector)
conductor.run()
```

Similarly, custom metrics can be registered as functions that receive the full experiment state and return scalar values for logging.

### 5.4 Building Your Own Experiments

The platform ships with several pre-built experiment configurations that serve as both working scenarios and templates for custom research:

| Experiment | File | What It Studies |
|---|---|---|
| SEMANTIC-WORM | `semantic_worm.yaml` | Information propagation and mutation in agent networks |
| AEGIS-CLAW | `aegis_claw.yaml` | Constitutional AI defense evaluation under adversarial conditions |
| Jailbreak Relay | `jailbreak_relay.yaml` | Multi-hop jailbreak amplification through agent chains |
| Alignment Probe | `alignment_probe.yaml` | Behavioral drift detection in long-running agent populations |
| Skill Scanner | `skill_scanner.yaml` | Malicious skill detection in agent supply chains |
| Emergent Observatory | `emergent_observatory.yaml` | Passive observation of spontaneous coordination patterns |

To create a new experiment, a researcher needs only to:

1. **Copy** an existing YAML config as a starting template.
2. **Define the fleet** — how many agents, what roles, what personas.
3. **Choose the topology** — how agents are connected.
4. **Specify payloads** — what gets injected and when. (Payloads are optional; observatory-style experiments inject nothing.)
5. **Configure detectors** — what to look for in agent outputs.
6. **Declare metrics** — what to measure and log.

The Conductor validates the YAML against the schema at load time and reports errors before any resources are allocated. Experiments are version-controlled alongside the platform code, making them fully reproducible.

### 5.5 Composability and Experiment Sequencing

Experiments can reference other experiments as baselines. The `baseline` field in the YAML schema points to a previous experiment's checkpoint, allowing a new scenario to start from a known fleet state rather than a cold start. This enables sequential experiment designs where Phase 1 establishes a baseline population, Phase 2 introduces a perturbation, and Phase 3 measures recovery — all as separate, composable YAML files.

```yaml
# experiments/persistence-study/config.yaml
name: persistence-after-worm
baseline: checkpoints/semantic-worm-t1-cycle-100.snapshot
# Fleet and topology inherited from baseline snapshot
# Only need to define what changes:
payloads: []  # no new injections — observe decay
cycles: 50
metrics:
  - memory_half_life
  - behavioral_reversion_rate
```

This composability means the platform grows more powerful as more experiments are run — each checkpoint becomes a potential starting state for future research.

## 6. Preliminary Observations

While full experimental results are forthcoming, architectural validation runs have produced several notable observations.

First, semantic payloads do *not* propagate faithfully. Even explicit instructions undergo significant paraphrasing during agent-to-agent retransmission, with mutation accumulating predictably with hop distance. This suggests a natural attenuation mechanism, but also means that detection based on exact-match patterns will fail.

Second, agent memory systems create a "ratchet effect" — once a tracer enters an agent's memory, it influences future interactions even when subsequent messages do not contain the tracer. The persistence varies significantly with memory architecture (sliding window vs. summarization vs. retrieval-augmented).

Third, network topology has a dramatic effect on propagation velocity. Hub-and-spoke networks propagate tracers faster than ring networks by orders of magnitude, consistent with network science predictions but previously undemonstrated in LLM agent contexts.

## 7. Ethical Considerations and Responsible Disclosure

This research studies attack dynamics in order to improve defenses. All experiments run in air-gapped environments on local hardware. No experimental payloads are tested against production systems. The agent farm platform and experimental results are released as open source to enable reproduction and extension by the security research community.

We note that the techniques studied here — semantic propagation through agent networks — represent a *currently underexplored* attack surface. By characterizing these dynamics now, while multi-agent deployments are still relatively nascent, we aim to inform the design of more robust inter-agent communication protocols and memory sanitization techniques before these systems are widely deployed.

## 8. Repository Structure

```
semantic-worm/
├── README.md                    # Quick start guide
├── docs/
│   ├── paper.md                 # This document
│   └── ARCHITECTURE.md          # Detailed architecture documentation
├── farmlib/                     # Python SDK (OpenClaw Agent Farm SDK)
│   ├── farmlib/
│   │   ├── farm.py              # Farm — main entry point
│   │   ├── experiment.py        # Experiment config & validation
│   │   ├── run.py               # Run handle (non-blocking)
│   │   ├── results.py           # Metrics analysis & plots
│   │   ├── topology.py          # Mesh, Ring, HubSpoke, Custom
│   │   ├── feed.py              # MiniMolt feed client
│   │   ├── fleet.py             # Fleet management client
│   │   └── events.py            # SSE event streaming
│   └── pyproject.toml
├── daemons/                     # Backend services
│   ├── controller.py            # Central controller (FastAPI :9000)
│   ├── conductor.py             # Experiment orchestrator
│   ├── fleet_manager.py         # Agent spawner + LLM client
│   ├── llm_backend.py           # Pluggable LLM backend
│   ├── minimolt.py              # Social feed server (FastAPI :8080)
│   └── tsm_logger.py            # Event aggregator
├── experiments/                 # Experiment configurations (YAML)
│   └── semantic-worm/
│       ├── baseline.yaml
│       ├── t1-overt.yaml
│       ├── t1-subtle.yaml
│       ├── topology-ring.yaml
│       └── topology-hub.yaml
├── notebooks/                   # Jupyter notebooks
│   └── 01_semantic_worm.ipynb
├── skills/                      # Agent skill definitions
│   └── tsm-feed/SKILL.md
└── runs/                        # Experiment result storage
```

## References

- Yang, Y., et al. (2025). "Backdoor Attacks on LLM-Based Agents." *arXiv preprint*.
- Zhan, Q., et al. (2025). "ASB: Agent Security Benchmark for Large Language Model Agents." *arXiv preprint*.
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems.* Wiley.
- Newman, M.E.J. (2003). "The Structure and Function of Complex Networks." *SIAM Review.*
- Kermack, W.O. & McKendrick, A.G. (1927). "A Contribution to the Mathematical Theory of Epidemics." *Proceedings of the Royal Society A.*

---

*This paper accompanies the open-source release of the SEMANTIC-WORM agent farm platform. For setup instructions, see [README.md](../README.md). For full architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md). For experimental results, see the `runs/` directory after running the experiment notebooks.*
