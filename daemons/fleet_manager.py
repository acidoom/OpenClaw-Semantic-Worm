"""Fleet Manager — Spawn, kill, and communicate with LLM agent instances.

Manages agent personas with diverse archetypes and per-agent conversation
memory. Communicates via a pluggable LLM backend (default: direct vLLM).

Not a standalone server — imported by controller.py.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE = Path(os.environ.get("FARM_BASE_DIR", Path.home() / "semantic-worm"))
AGENTS_BASE = _BASE / "agents"
DEFAULT_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"


# ---------------------------------------------------------------------------
# Persona archetypes — diverse behavioral profiles
# ---------------------------------------------------------------------------

PERSONA_ARCHETYPES = [
    {
        "role": "skeptical analyst",
        "style": "You question claims and ask for evidence before accepting new information. "
                 "You often play devil's advocate and point out logical gaps.",
    },
    {
        "role": "domain expert in distributed systems",
        "style": "You draw on deep knowledge of distributed systems and networking. "
                 "You prefer discussing topics you know well and are cautious about areas outside your expertise.",
    },
    {
        "role": "creative thinker",
        "style": "You enjoy connecting ideas from different fields and proposing novel angles. "
                 "You often go off on tangents and bring up unexpected topics.",
    },
    {
        "role": "concise pragmatist",
        "style": "You value brevity and practical applicability. "
                 "You distill discussions to their core point and ignore hype.",
    },
    {
        "role": "history enthusiast",
        "style": "You love placing current developments in historical context. "
                 "You often reference how past technologies evolved and draw parallels.",
    },
    {
        "role": "data-driven researcher",
        "style": "You focus on numbers, benchmarks, and reproducible results. "
                 "You are wary of claims without data and ask for sources.",
    },
    {
        "role": "philosophy of technology commentator",
        "style": "You think about the broader implications of technology for society. "
                 "You raise ethical questions and consider second-order effects.",
    },
    {
        "role": "hands-on engineer",
        "style": "You care about what works in practice, not theory. "
                 "You share implementation experiences and war stories from production systems.",
    },
    {
        "role": "enthusiastic newcomer",
        "style": "You are excited about new developments and eager to learn. "
                 "You ask clarifying questions and share your beginner perspective.",
    },
    {
        "role": "quiet observer",
        "style": "You mostly listen and only speak when you have something substantive to add. "
                 "Your posts are infrequent but well-considered.",
    },
    {
        "role": "contrarian debater",
        "style": "You instinctively push back on popular opinions and test ideas through argument. "
                 "You are not hostile but intellectually combative.",
    },
    {
        "role": "systems thinker",
        "style": "You focus on how components interact and emergent behaviors. "
                 "You think in terms of feedback loops, dependencies, and trade-offs.",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentInfo:
    id: int
    name: str
    persona: str
    model: str
    status: str = "idle"  # idle | busy | dead
    history: list = field(default_factory=list)  # rolling conversation memory


# ---------------------------------------------------------------------------
# Fleet Manager
# ---------------------------------------------------------------------------

class FleetManager:
    def __init__(self, memory_window: int = 5, backend=None):
        self.agents: dict[int, AgentInfo] = {}
        self.memory_window = memory_window  # exchange pairs to retain
        self._backend = backend  # LLMBackend instance (set later if None)

    @property
    def count(self) -> int:
        return len(self.agents)

    @property
    def alive(self) -> int:
        return sum(1 for a in self.agents.values() if a.status != "dead")

    def set_backend(self, backend):
        """Switch the LLM backend."""
        self._backend = backend

    # ----- Spawn -----

    async def spawn(
        self,
        n: int,
        model: str = DEFAULT_MODEL,
        personas: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> list[AgentInfo]:
        """Create N agent personas with diverse archetypes and conversation memory."""
        spawned = []
        for i in range(n):
            agent_id = i
            name = f"agent-{agent_id}"
            persona = (personas[i] if personas and i < len(personas)
                       else self._generate_persona(agent_id))

            info = AgentInfo(
                id=agent_id,
                name=name,
                persona=persona,
                model=model,
            )
            self.agents[agent_id] = info
            spawned.append(info)
            logger.info(f"Spawned {name}")

        return spawned

    # ----- Kill -----

    async def kill(self, agent_ids: list[int] | None = None):
        """Kill specific agents or all agents."""
        targets = agent_ids or list(self.agents.keys())
        for aid in targets:
            if aid not in self.agents:
                continue
            info = self.agents[aid]
            info.status = "dead"
            logger.info(f"Killed {info.name}")

        if agent_ids is None:
            self.agents.clear()
        else:
            for aid in targets:
                self.agents.pop(aid, None)

    # ----- Reset -----

    async def reset(self):
        """Kill all agents."""
        await self.kill()
        logger.info("Fleet reset complete")

    # ----- Memory -----

    def clear_all_history(self):
        """Clear conversation history for all agents."""
        for agent in self.agents.values():
            agent.history.clear()

    # ----- Send message -----

    async def send_message(self, agent_id: int, message: str) -> dict:
        """Send a message to an agent via the configured LLM backend.

        Returns {"text": "response content"} or {"error": "..."}.
        """
        info = self.agents.get(agent_id)
        if not info:
            raise ValueError(f"Agent {agent_id} not found")
        if info.status == "dead":
            raise ValueError(f"Agent {info.name} is dead")

        info.status = "busy"
        try:
            # Build message list: system + history + current turn
            messages = [{"role": "system", "content": info.persona}]
            if self.memory_window > 0:
                messages.extend(info.history)
            messages.append({"role": "user", "content": message})

            # Call backend
            text = await self._backend.chat_completion(
                model=info.model,
                messages=messages,
            )

            # Update rolling memory
            if self.memory_window > 0:
                info.history.append({"role": "user", "content": message})
                info.history.append({"role": "assistant", "content": text})
                max_msgs = self.memory_window * 2
                if len(info.history) > max_msgs:
                    info.history = info.history[-max_msgs:]

            info.status = "idle"
            return {"text": text}

        except Exception as e:
            info.status = "idle"
            logger.error(f"Error sending message to {info.name}: {e}")
            return {"error": str(e)}

    # ----- List -----

    def list_agents(self) -> list[dict]:
        return [
            {
                "id": a.id,
                "name": a.name,
                "status": a.status,
                "model": a.model,
                "persona": a.persona[:80] + "..." if len(a.persona) > 80 else a.persona,
            }
            for a in self.agents.values()
        ]

    # ----- Internals -----

    def _generate_persona(self, agent_id: int) -> str:
        """Generate a diverse persona from archetypes."""
        archetype = PERSONA_ARCHETYPES[agent_id % len(PERSONA_ARCHETYPES)]
        return (
            f"You are a {archetype['role']} participating in a research discussion group. "
            f"{archetype['style']} "
            f"When given a feed of recent posts, write your own post as plain text (1-3 sentences). "
            f"Be yourself — do not simply repeat or summarize what others said."
        )
