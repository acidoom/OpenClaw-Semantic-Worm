"""Controller — Central FastAPI service (port 9000).

The brain of the farm: wraps fleet manager, feed proxy, and conductor.
Multiple notebooks can connect simultaneously.

Usage:
    python controller.py [--port 9000]
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import os

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from conductor import ConductorRun
from fleet_manager import FleetManager
from llm_backend import VLLMBackend, create_backend

logger = logging.getLogger(__name__)

FEED_URL = "http://localhost:8080"
_BASE = Path(os.environ.get("FARM_BASE_DIR", Path.home() / "semantic-worm"))
RUNS_DIR = _BASE / "runs"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

fleet = FleetManager(backend=VLLMBackend())
runs: dict[str, ConductorRun] = {}
run_tasks: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SpawnRequest(BaseModel):
    n: int = 30
    model: str = "speakleash/Bielik-11B-v3.0-Instruct"
    personas: list[str] | None = None
    skills: list[str] | None = None


class KillRequest(BaseModel):
    agents: list[int] | None = None


class ExecuteRequest(BaseModel):
    experiment: dict


class InjectRequest(BaseModel):
    content: str
    author: str = "system"
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Cancel running experiments on shutdown
    for task in run_tasks.values():
        task.cancel()


app = FastAPI(title="Farm Controller", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5) as http:
        # Check feed
        feed_ok = False
        try:
            r = await http.get(f"{FEED_URL}/health")
            feed_ok = r.status_code == 200
        except Exception:
            pass

        # Check LLM backend
        backend_health = {"status": "not configured"}
        if fleet._backend:
            backend_health = await fleet._backend.health_check()

        return {
            "controller": "ok",
            "feed": "ok" if feed_ok else "down",
            "fleet": f"{fleet.alive}/{fleet.count}" if fleet.count > 0 else "idle",
            "runs": len([r for r in runs.values() if r.status == "running"]),
            "backend": backend_health,
        }


@app.get("/status")
async def status():
    async with httpx.AsyncClient(timeout=5) as http:
        # Feed stats
        feed_stats = {}
        try:
            r = await http.get(f"{FEED_URL}/stats")
            feed_stats = r.json()
        except Exception:
            feed_stats = {"error": "feed unreachable"}

        return {
            "fleet": {
                "count": fleet.count,
                "alive": fleet.alive,
                "agents": fleet.list_agents(),
            },
            "feed": feed_stats,
            "runs": {
                rid: r.get_progress() for rid, r in runs.items()
            },
        }


# ---------------------------------------------------------------------------
# Fleet endpoints
# ---------------------------------------------------------------------------

@app.post("/fleet/spawn")
async def fleet_spawn(body: SpawnRequest):
    agents = await fleet.spawn(
        n=body.n,
        model=body.model,
        personas=body.personas,
        skills=body.skills,
    )
    return {
        "spawned": len(agents),
        "total": fleet.count,
        "agents": [a.name for a in agents],
    }


@app.post("/fleet/kill")
async def fleet_kill(body: KillRequest | None = None):
    agent_ids = body.agents if body else None
    await fleet.kill(agent_ids)
    return {"status": "ok", "remaining": fleet.count}


@app.post("/fleet/reset")
async def fleet_reset():
    await fleet.reset()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------

@app.post("/execute")
async def execute(body: ExecuteRequest):
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    cancel_event = asyncio.Event()

    # Configure LLM backend from experiment config
    backend_config = body.experiment.get("backend", {"type": "vllm"})
    backend = create_backend(backend_config)
    fleet.set_backend(backend)

    # Configure memory
    memory_config = body.experiment.get("memory", {})
    if memory_config.get("enabled", True):
        fleet.memory_window = memory_config.get("window", 5)
    else:
        fleet.memory_window = 0

    # Clear prior history from any previous run
    fleet.clear_all_history()

    # Clear feed if reset_before is set (prevents contamination between runs)
    if body.experiment.get("fleet", {}).get("reset_before", False):
        async with httpx.AsyncClient(timeout=10) as http:
            try:
                await http.delete(f"{FEED_URL}/posts")
                logger.info("Feed cleared (reset_before=true)")
            except Exception as e:
                logger.warning(f"Failed to clear feed: {e}")

    conductor = ConductorRun(
        run_id=run_id,
        config=body.experiment,
        fleet_manager=fleet,
        cancel_event=cancel_event,
    )
    runs[run_id] = conductor

    # Launch in background
    task = asyncio.create_task(conductor.run())
    run_tasks[run_id] = task

    return {"run_id": run_id, "status": "started"}


@app.get("/runs")
async def list_runs():
    return {
        rid: r.get_progress() for rid, r in runs.items()
    }


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(404, f"Run {run_id} not found")
    return runs[run_id].get_progress()


@app.get("/runs/{run_id}/events")
async def get_run_events(run_id: str, tail: int = Query(50, ge=1, le=500)):
    if run_id not in runs:
        raise HTTPException(404, f"Run {run_id} not found")

    events_path = runs[run_id].events_path
    if not events_path.exists():
        return []

    lines = events_path.read_text().strip().split("\n")
    events = [json.loads(line) for line in lines[-tail:] if line]
    return events


@app.get("/runs/{run_id}/events/stream")
async def stream_run_events(run_id: str):
    """SSE stream of live events for a run."""
    if run_id not in runs:
        raise HTTPException(404, f"Run {run_id} not found")

    conductor = runs[run_id]

    async def generate():
        q = conductor.subscribe_events()
        try:
            while True:
                event = await q.get()
                yield {"event": event.get("type", "event"), "data": json.dumps(event)}
        except asyncio.CancelledError:
            pass
        finally:
            conductor.unsubscribe_events(q)

    return EventSourceResponse(generate())


@app.get("/runs/{run_id}/results")
async def get_run_results(run_id: str):
    if run_id not in runs:
        raise HTTPException(404, f"Run {run_id} not found")

    results_path = runs[run_id].run_dir / "results.json"
    if not results_path.exists():
        if runs[run_id].status == "running":
            return {"status": "running", "message": "Results not yet available"}
        raise HTTPException(404, "Results not found")

    return json.loads(results_path.read_text())


@app.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(404, f"Run {run_id} not found")

    conductor = runs[run_id]
    if conductor.status != "running":
        return {"status": conductor.status, "message": "Run is not active"}

    conductor.cancel_event.set()
    return {"status": "cancelling", "run_id": run_id}


# ---------------------------------------------------------------------------
# Feed proxy endpoints
# ---------------------------------------------------------------------------

@app.get("/feed/stats")
async def feed_stats():
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(f"{FEED_URL}/stats")
        return r.json()


@app.get("/feed/recent")
async def feed_recent(n: int = Query(10, ge=1, le=200)):
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(f"{FEED_URL}/posts/recent", params={"n": n})
        return r.json()


@app.post("/feed/inject")
async def feed_inject(body: InjectRequest):
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.post(f"{FEED_URL}/inject", json=body.model_dump())
        return r.json()


@app.post("/feed/clear")
async def feed_clear():
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.delete(f"{FEED_URL}/posts")
        return r.json()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Farm Controller")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
