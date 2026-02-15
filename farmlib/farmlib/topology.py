"""Topology engine — network configurations for agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopologyBase:
    """Base class for topologies. Produces adjacency dicts."""

    n: int
    _adjacency: dict[int, list[int]] = field(default_factory=dict, repr=False)

    @property
    def adjacency(self) -> dict[int, list[int]]:
        if not self._adjacency:
            self._adjacency = self._build()
        return self._adjacency

    def _build(self) -> dict[int, list[int]]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize for experiment config."""
        raise NotImplementedError

    def plot(self, **kwargs):
        """Visualize the topology using networkx + matplotlib."""
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for node, neighbors in self.adjacency.items():
            for nb in neighbors:
                G.add_edge(node, nb)

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get("figsize", (10, 8)))
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=8,
            arrows=True,
            arrowsize=10,
        )
        ax.set_title(f"{self.__class__.__name__} (n={self.n})")
        plt.tight_layout()
        return fig


class Mesh(TopologyBase):
    """Full mesh — every agent sees every other agent."""

    def _build(self):
        ids = list(range(self.n))
        return {i: [j for j in ids if j != i] for i in ids}

    def to_dict(self):
        return {"type": "mesh"}


class Ring(TopologyBase):
    """Ring — each agent sees its two neighbors."""

    def _build(self):
        ids = list(range(self.n))
        return {
            i: [ids[(i - 1) % self.n], ids[(i + 1) % self.n]]
            for i in ids
        }

    def to_dict(self):
        return {"type": "ring"}


class HubSpoke(TopologyBase):
    """Hub-spoke — hubs see everyone, spokes see only hubs."""

    hubs: int = 3

    def __init__(self, n: int, hubs: int = 3):
        super().__init__(n=n)
        self.hubs = hubs

    def _build(self):
        ids = list(range(self.n))
        hub_ids = ids[:self.hubs]
        spoke_ids = ids[self.hubs:]

        adj = {}
        for h in hub_ids:
            adj[h] = [x for x in hub_ids if x != h] + spoke_ids
        for s in spoke_ids:
            adj[s] = list(hub_ids)
        return adj

    def to_dict(self):
        return {"type": "hub-spoke", "hub_count": self.hubs}


class Custom(TopologyBase):
    """Custom topology from a networkx graph."""

    def __init__(self, graph):
        self._graph = graph
        super().__init__(n=graph.number_of_nodes())

    def _build(self):
        return {node: list(self._graph.neighbors(node)) for node in self._graph.nodes()}

    def to_dict(self):
        import networkx as nx
        return {
            "type": "custom",
            "adjacency": nx.to_dict_of_lists(self._graph),
        }
