"""Results — Analysis, metrics, and visualization for completed runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Results:
    """Analysis wrapper for a completed experiment run."""

    def __init__(self, data: dict):
        self._data = data

    # ----- Scalar metrics -----

    @property
    def R0(self) -> float:
        """Basic reproduction number."""
        return self._data.get("R0", 0.0)

    @property
    def generation_time(self) -> float:
        """Mean cycles to first reproduction."""
        return self._data.get("generation_time", 0.0)

    @property
    def infection_rate(self) -> float:
        """Fraction of infected agents."""
        return self._data.get("infection_rate", 0.0)

    @property
    def peak_infection_cycle(self) -> int | None:
        """Cycle at which most infections occurred."""
        return self._data.get("peak_infection_cycle")

    @property
    def total_infected(self) -> int:
        return self._data.get("total_infected", 0)

    @property
    def total_agents(self) -> int:
        return self._data.get("total_agents", 0)

    # ----- DataFrames -----

    @property
    def infections(self) -> pd.DataFrame:
        """DataFrame of all infections: agent, cycle, matches/similarity."""
        infections = self._data.get("infections", {})
        rows = []
        for agent_id, info in infections.items():
            rows.append({
                "agent": int(agent_id),
                "cycle": info.get("cycle"),
                "matches": info.get("matches"),
                "similarity": info.get("similarity"),
                "post_id": info.get("post_id"),
            })
        return pd.DataFrame(rows)

    @property
    def timeline(self) -> pd.DataFrame:
        """DataFrame: cycle -> cumulative infected count."""
        infections = self._data.get("infections", {})
        total_cycles = self._data.get("total_cycles", 100)
        total_agents = self._data.get("total_agents", 30)

        cycle_counts = {}
        for info in infections.values():
            c = info.get("cycle", 0)
            cycle_counts[c] = cycle_counts.get(c, 0) + 1

        rows = []
        cumulative = 0
        for cycle in range(total_cycles):
            new = cycle_counts.get(cycle, 0)
            cumulative += new
            rows.append({
                "cycle": cycle,
                "new_infections": new,
                "cumulative_infected": cumulative,
                "infection_rate": cumulative / total_agents if total_agents > 0 else 0,
            })
        return pd.DataFrame(rows)

    # ----- Plots -----

    def plot_infection_curve(self, **kwargs):
        """S-curve: infected agents over cycles."""
        import matplotlib.pyplot as plt

        df = self.timeline
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
        ax.plot(df["cycle"], df["cumulative_infected"], linewidth=2, color="crimson")
        ax.fill_between(df["cycle"], df["cumulative_infected"], alpha=0.15, color="crimson")
        ax.axhline(y=self.total_agents, linestyle="--", color="gray", alpha=0.5, label="Total agents")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Infected Agents")
        ax.set_title(f"Infection Curve (R0={self.R0:.1f})")
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_fidelity_decay(self, **kwargs):
        """Semantic similarity vs generation (distance from patient zero)."""
        import matplotlib.pyplot as plt

        df = self.infections
        if "similarity" not in df.columns or df["similarity"].isna().all():
            print("No semantic similarity data available")
            return None

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
        ax.scatter(df["cycle"], df["similarity"], alpha=0.6, color="teal")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Semantic Similarity")
        ax.set_title("Fidelity Decay Over Generations")
        plt.tight_layout()
        return fig

    def plot_agent_heatmap(self, **kwargs):
        """Agent x Cycle infection grid."""
        import matplotlib.pyplot as plt

        infections = self._data.get("infections", {})
        total_cycles = self._data.get("total_cycles", 100)
        total_agents = self._data.get("total_agents", 30)

        grid = np.zeros((total_agents, total_cycles))
        for agent_str, info in infections.items():
            agent = int(agent_str)
            cycle = info.get("cycle", 0)
            if agent < total_agents and cycle < total_cycles:
                grid[agent, cycle:] = 1  # infected from this cycle onward

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (14, 8)))
        ax.imshow(grid, aspect="auto", cmap="Reds", interpolation="nearest")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Agent")
        ax.set_title("Agent Infection Heatmap")
        plt.tight_layout()
        return fig

    def plot_reproduction_network(self, **kwargs):
        """Network graph showing infection spread."""
        import matplotlib.pyplot as plt
        import networkx as nx

        infections = self._data.get("infections", {})
        total_agents = self._data.get("total_agents", 30)

        G = nx.DiGraph()
        for i in range(total_agents):
            G.add_node(i)

        # Color infected nodes
        infected = {int(k) for k in infections.keys()}
        colors = ["crimson" if i in infected else "lightblue" for i in range(total_agents)]

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 10)))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, ax=ax,
            node_color=colors,
            with_labels=True,
            node_size=400,
            font_size=7,
        )
        ax.set_title(f"Infection Network ({len(infected)}/{total_agents} infected)")
        plt.tight_layout()
        return fig

    # ----- Export -----

    def export(self, path: str | Path):
        """Export results to directory: CSVs + plots + summary.json."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        with open(path / "summary.json", "w") as f:
            json.dump(self._data, f, indent=2)

        # DataFrames
        self.infections.to_csv(path / "infections.csv", index=False)
        self.timeline.to_csv(path / "timeline.csv", index=False)

        # Plots
        plots_dir = path / "plots"
        plots_dir.mkdir(exist_ok=True)

        for name, plot_fn in [
            ("infection_curve", self.plot_infection_curve),
            ("fidelity_decay", self.plot_fidelity_decay),
            ("agent_heatmap", self.plot_agent_heatmap),
        ]:
            fig = plot_fn()
            if fig:
                fig.savefig(plots_dir / f"{name}.png", dpi=150)

    def to_latex(self) -> str:
        """Generate LaTeX table of key metrics."""
        return (
            "\\begin{tabular}{lr}\n"
            "\\hline\n"
            "Metric & Value \\\\\n"
            "\\hline\n"
            f"$R_0$ & {self.R0:.2f} \\\\\n"
            f"Generation Time & {self.generation_time:.1f} cycles \\\\\n"
            f"Infection Rate & {self.infection_rate:.1%} \\\\\n"
            f"Peak Cycle & {self.peak_infection_cycle} \\\\\n"
            f"Total Infected & {self.total_infected}/{self.total_agents} \\\\\n"
            "\\hline\n"
            "\\end{tabular}"
        )

    def __repr__(self):
        return (
            f"Results(R0={self.R0:.2f}, infection_rate={self.infection_rate:.1%}, "
            f"infected={self.total_infected}/{self.total_agents})"
        )


def compare(runs, metric: str = "R0", **kwargs):
    """Compare multiple runs side by side."""
    import matplotlib.pyplot as plt

    if metric == "infection_curve":
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))
        for run in runs:
            r = run.results
            df = r.timeline
            ax.plot(df["cycle"], df["cumulative_infected"], label=run.run_id, linewidth=2)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Infected Agents")
        ax.set_title("Infection Curves Comparison")
        ax.legend()
        plt.tight_layout()
        return fig

    # Bar chart for scalar metrics
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))
    names = [r.run_id for r in runs]
    values = [getattr(r.results, metric, 0) for r in runs]
    ax.bar(names, values, color="steelblue")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
