"""TSM Logger — Event aggregator.

Connects to the controller's SSE event streams and writes events
to a combined JSONL log and optional SQLite database for queries.

Usage:
    python tsm_logger.py [--controller http://localhost:9000]
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

import os

CONTROLLER_URL = "http://localhost:9000"
_BASE = Path(os.environ.get("FARM_BASE_DIR", Path.home() / "semantic-worm"))
LOG_DIR = _BASE / "logs"
DB_PATH = LOG_DIR / "events.db"
JSONL_PATH = LOG_DIR / "events.jsonl"


class TSMLogger:
    def __init__(self, controller_url: str = CONTROLLER_URL):
        self.controller_url = controller_url
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        self.db = sqlite3.connect(str(DB_PATH))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                type TEXT,
                cycle INTEGER,
                agent INTEGER,
                data TEXT,
                timestamp TEXT
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
        self.db.commit()

    def log_event(self, event: dict):
        """Write event to JSONL and SQLite."""
        # JSONL
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")

        # SQLite
        self.db.execute(
            "INSERT INTO events (run_id, type, cycle, agent, data, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.get("run_id"),
                event.get("type"),
                event.get("cycle"),
                event.get("agent"),
                json.dumps(event),
                event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ),
        )
        self.db.commit()

    async def watch_run(self, run_id: str):
        """Subscribe to a run's event stream and log all events."""
        url = f"{self.controller_url}/runs/{run_id}/events/stream"
        logger.info(f"Watching run {run_id} at {url}")

        async with httpx.AsyncClient(timeout=None) as http:
            async with http.stream("GET", url) as response:
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)
                        for line in message.split("\n"):
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                try:
                                    event = json.loads(data)
                                    self.log_event(event)
                                    logger.info(
                                        f"[{event.get('type')}] cycle={event.get('cycle')} "
                                        f"agent={event.get('agent')}"
                                    )
                                except json.JSONDecodeError:
                                    pass

    async def poll_runs(self):
        """Poll controller for active runs and watch them."""
        watched: set[str] = set()
        while True:
            try:
                async with httpx.AsyncClient(timeout=10) as http:
                    r = await http.get(f"{self.controller_url}/runs")
                    all_runs = r.json()

                    for run_id, info in all_runs.items():
                        if info.get("status") == "running" and run_id not in watched:
                            watched.add(run_id)
                            asyncio.create_task(self.watch_run(run_id))
                            logger.info(f"Started watching {run_id}")
            except Exception as e:
                logger.warning(f"Poll error: {e}")

            await asyncio.sleep(5)

    def query_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query logged events from SQLite."""
        conditions = []
        params = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if event_type:
            conditions.append("type = ?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = self.db.execute(
            f"SELECT data FROM events {where} ORDER BY id DESC LIMIT ?",
            params + [limit],
        )
        return [json.loads(row[0]) for row in cursor.fetchall()]


async def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="TSM Event Logger")
    parser.add_argument("--controller", default=CONTROLLER_URL)
    args = parser.parse_args()

    tsm = TSMLogger(args.controller)
    await tsm.poll_runs()


if __name__ == "__main__":
    asyncio.run(main())
