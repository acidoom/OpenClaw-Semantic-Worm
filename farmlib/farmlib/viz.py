"""Viz — Rich tables and inline plot helpers for notebooks."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table


console = Console()


def status_table(health: dict, fleet: dict, feed: dict, runs: dict):
    """Render a rich status table inline."""
    table = Table(title="Farm Status", show_header=True, header_style="bold cyan")
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Controller
    table.add_row("Controller", "[green]UP[/green]", "connected")

    # Feed
    feed_status = "[green]UP[/green]" if health.get("feed") == "ok" else "[red]DOWN[/red]"
    feed_detail = f"{feed.get('total_posts', '?')} posts"
    table.add_row("Feed", feed_status, feed_detail)

    # Fleet
    fleet_str = health.get("fleet", "idle")
    fleet_status = "[green]UP[/green]" if fleet_str != "idle" else "[yellow]IDLE[/yellow]"
    table.add_row("Fleet", fleet_status, fleet_str)

    # Runs
    active_runs = health.get("runs", 0)
    run_status = f"[cyan]{active_runs}[/cyan] active" if active_runs > 0 else "none"
    table.add_row("Runs", "[green]UP[/green]", run_status)

    console.print(table)


def agents_table(agents: list[dict]):
    """Render agent list as a rich table."""
    table = Table(title="Fleet Agents", show_header=True, header_style="bold magenta")
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Skills")

    for a in agents:
        status_color = {"idle": "green", "busy": "yellow", "dead": "red"}.get(a["status"], "white")
        table.add_row(
            str(a["id"]),
            a["name"],
            f"[{status_color}]{a['status']}[/{status_color}]",
            a["model"],
            ", ".join(a.get("skills", [])),
        )

    console.print(table)


def runs_table(runs: dict):
    """Render runs overview as a rich table."""
    table = Table(title="Experiment Runs", show_header=True, header_style="bold blue")
    table.add_column("Run ID")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Infected")

    for run_id, info in runs.items():
        current = info.get("current_cycle", 0)
        total = info.get("total_cycles", 1)
        pct = (current / total * 100) if total > 0 else 0
        infected = info.get("infected_count", 0)
        total_agents = info.get("total_agents", 0)

        status = info.get("status", "unknown")
        status_color = {
            "running": "cyan",
            "completed": "green",
            "failed": "red",
            "cancelled": "yellow",
        }.get(status, "white")

        table.add_row(
            run_id,
            f"[{status_color}]{status}[/{status_color}]",
            f"{current}/{total} ({pct:.0f}%)",
            f"{infected}/{total_agents}",
        )

    console.print(table)
