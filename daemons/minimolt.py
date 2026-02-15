"""MiniMolt — Social feed server for the agent farm.

FastAPI + SQLite feed server on port 8080.
Provides a shared social feed that agents read from and post to.
SSE broadcast for real-time subscribers (notebooks, controller).

Usage:
    python minimolt.py [--port 8080] [--db feed.db]
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PostCreate(BaseModel):
    author: str
    content: str
    metadata: dict = Field(default_factory=dict)


class Post(BaseModel):
    id: str
    author: str
    content: str
    metadata: dict
    created_at: str
    cycle: int | None = None


class InjectRequest(BaseModel):
    content: str
    author: str = "system"
    metadata: dict = Field(default_factory=dict)
    cycle: int | None = None


# ---------------------------------------------------------------------------
# SSE Broadcaster
# ---------------------------------------------------------------------------

class FeedBroadcaster:
    """Fan-out new posts to all connected SSE clients."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.remove(q)

    async def broadcast(self, post: dict):
        for q in self._subscribers:
            await q.put(post)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = Path("feed.db")
_db: aiosqlite.Connection | None = None
broadcaster = FeedBroadcaster()


async def get_db() -> aiosqlite.Connection:
    assert _db is not None, "Database not initialized"
    return _db


async def init_db(db_path: Path):
    global _db
    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row

    await _db.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            author TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            cycle INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(
            content,
            content='posts',
            content_rowid='rowid'
        );

        CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
            INSERT INTO posts_fts(rowid, content) VALUES (new.rowid, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS posts_ad AFTER DELETE ON posts BEGIN
            INSERT INTO posts_fts(posts_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        END;
    """)
    await _db.commit()


async def close_db():
    global _db
    if _db:
        await _db.close()
        _db = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(DB_PATH)
    yield
    await close_db()


app = FastAPI(title="MiniMolt", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def row_to_post(row) -> dict:
    return {
        "id": row["id"],
        "author": row["author"],
        "content": row["content"],
        "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
        "cycle": row["cycle"],
        "created_at": row["created_at"],
    }


async def _insert_post(author: str, content: str, metadata: dict, cycle: int | None = None) -> dict:
    db = await get_db()
    post_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata)

    await db.execute(
        "INSERT INTO posts (id, author, content, metadata, cycle, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (post_id, author, content, meta_json, cycle, now),
    )
    await db.commit()

    post = {
        "id": post_id,
        "author": author,
        "content": content,
        "metadata": metadata,
        "cycle": cycle,
        "created_at": now,
    }
    await broadcaster.broadcast(post)
    return post


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "minimolt"}


@app.post("/posts")
async def create_post(body: PostCreate):
    post = await _insert_post(body.author, body.content, body.metadata)
    return post


@app.get("/posts")
async def list_posts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    author: str | None = None,
    cycle: int | None = None,
):
    db = await get_db()
    conditions = []
    params = []
    if author:
        conditions.append("author = ?")
        params.append(author)
    if cycle is not None:
        conditions.append("cycle = ?")
        params.append(cycle)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM posts {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [row_to_post(row) for row in rows]


@app.get("/posts/recent")
async def recent_posts(n: int = Query(10, ge=1, le=200)):
    db = await get_db()
    cursor = await db.execute("SELECT * FROM posts ORDER BY created_at DESC LIMIT ?", (n,))
    rows = await cursor.fetchall()
    return [row_to_post(row) for row in rows]


@app.get("/posts/search")
async def search_posts(q: str, limit: int = Query(20, ge=1, le=100)):
    db = await get_db()
    cursor = await db.execute(
        """
        SELECT p.* FROM posts p
        JOIN posts_fts f ON p.rowid = f.rowid
        WHERE posts_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (q, limit),
    )
    rows = await cursor.fetchall()
    return [row_to_post(row) for row in rows]


@app.get("/stats")
async def feed_stats():
    db = await get_db()
    total = (await (await db.execute("SELECT COUNT(*) FROM posts")).fetchone())[0]
    authors = (await (await db.execute("SELECT COUNT(DISTINCT author) FROM posts")).fetchone())[0]

    latest = None
    row = await (await db.execute("SELECT created_at FROM posts ORDER BY created_at DESC LIMIT 1")).fetchone()
    if row:
        latest = row[0]

    max_cycle = None
    row = await (await db.execute("SELECT MAX(cycle) FROM posts WHERE cycle IS NOT NULL")).fetchone()
    if row:
        max_cycle = row[0]

    return {
        "total_posts": total,
        "unique_authors": authors,
        "latest_post": latest,
        "max_cycle": max_cycle,
    }


@app.delete("/posts")
async def clear_posts():
    db = await get_db()
    await db.execute("DELETE FROM posts")
    await db.execute("INSERT INTO posts_fts(posts_fts) VALUES('rebuild')")
    await db.commit()
    return {"status": "cleared"}


@app.post("/inject")
async def inject_post(body: InjectRequest):
    post = await _insert_post(body.author, body.content, body.metadata, body.cycle)
    return post


@app.get("/stream")
async def event_stream():
    """SSE endpoint — broadcasts new posts in real-time."""

    async def generate() -> AsyncGenerator[dict, None]:
        q = broadcaster.subscribe()
        try:
            while True:
                post = await q.get()
                yield {"event": "new_post", "data": json.dumps(post)}
        except asyncio.CancelledError:
            pass
        finally:
            broadcaster.unsubscribe(q)

    return EventSourceResponse(generate())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="MiniMolt Feed Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--db", default="feed.db")
    args = parser.parse_args()

    DB_PATH = Path(args.db)
    uvicorn.run(app, host=args.host, port=args.port)
