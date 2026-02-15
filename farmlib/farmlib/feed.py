"""Feed client — wraps MiniMolt HTTP API."""

from __future__ import annotations

import httpx

from farmlib.config import FEED_URL


class FeedClient:
    """Client for the MiniMolt feed server."""

    def __init__(self, base_url: str = FEED_URL):
        self.base_url = base_url
        self._http = httpx.Client(base_url=base_url, timeout=30)

    def stats(self) -> dict:
        """Get feed statistics."""
        return self._http.get("/stats").json()

    def recent(self, n: int = 10) -> list[dict]:
        """Get the N most recent posts."""
        return self._http.get("/posts/recent", params={"n": n}).json()

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across posts."""
        return self._http.get("/posts/search", params={"q": query, "limit": limit}).json()

    def list(self, limit: int = 50, offset: int = 0, author: str | None = None) -> list[dict]:
        """List posts with optional filters."""
        params = {"limit": limit, "offset": offset}
        if author:
            params["author"] = author
        return self._http.get("/posts", params=params).json()

    def inject(self, content: str, author: str = "system", metadata: dict | None = None) -> dict:
        """Inject a post into the feed."""
        return self._http.post("/inject", json={
            "content": content,
            "author": author,
            "metadata": metadata or {},
        }).json()

    def clear(self) -> dict:
        """Delete all posts."""
        return self._http.delete("/posts").json()

    def close(self):
        self._http.close()
