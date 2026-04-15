import logging
import os
import sys
import time
from datetime import datetime

# ── Rich imports ───────────────────────────────────────────────────────────
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Singleton console ─────────────────────────────────────────────────────
console = Console(width=90)

# ── File logger ────────────────────────────────────────────────────────────
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = os.path.join(LOGS_DIR, f"pipeline_{_timestamp}.log")

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
)

_logger = logging.getLogger("ml_pipeline")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_file_handler)
# Suppress duplicate console output from logging
_logger.propagate = False


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def banner():
    """Display the main pipeline banner."""
    title = Text()
    title.append("🛡️  GLOBAL CYBERSECURITY THREATS (2015-2024)\n", style="bold cyan")
    title.append("Machine Learning Pipeline", style="bold white")
    panel = Panel(
        title,
        border_style="bright_cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
        subtitle=f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
    )
    console.print()
    console.print(panel, justify="center")
    _logger.info("Pipeline started")


def step_header(step_num: int, total: int, title: str, icon: str = "📌"):
    """Display a major step header."""
    console.print()
    console.print(Rule(style="bright_blue"))
    label = f"{icon}  STEP {step_num}/{total} — {title}"
    console.print(
        Panel(label, style="bold bright_blue", box=box.ROUNDED, padding=(0, 2)),
        justify="center",
    )
    console.print(Rule(style="bright_blue"))
    _logger.info(f"===== STEP {step_num}/{total}: {title} =====")


def section(title: str, icon: str = "─"):
    """Display a section divider within a step."""
    console.print()
    console.print(f"  [bold yellow]{icon}[/bold yellow] [bold]{title}[/bold]")
    _logger.info(f"--- {title} ---")


def info(msg: str):
    """Informational message."""
    console.print(f"  [dim]ℹ[/dim]  {msg}")
    _logger.info(msg)


def success(msg: str):
    """Success message."""
    console.print(f"  [green]✔[/green]  {msg}")
    _logger.info(f"[OK] {msg}")


def warning(msg: str):
    """Warning message."""
    console.print(f"  [yellow]⚠[/yellow]  [yellow]{msg}[/yellow]")
    _logger.warning(msg)


def error(msg: str):
    """Error message."""
    console.print(f"  [red]✖[/red]  [red]{msg}[/red]")
    _logger.error(msg)


def metric(name: str, value, fmt: str = ".4f"):
    """Display a single metric."""
    if isinstance(value, float):
        val_str = f"{value:{fmt}}"
    else:
        val_str = str(value)
    console.print(f"    [cyan]{name:<24}[/cyan] [bold white]{val_str}[/bold white]")
    _logger.info(f"  {name}: {val_str}")


def metrics_table(title: str, headers: list, rows: list, highlight_best: int = None):
    """Display a rich table of metrics.
    
    Args:
        title: Table title.
        headers: List of column header strings.
        rows: List of row tuples/lists.
        highlight_best: Column index (0-based) to highlight the best value (max).
    """
    table = Table(
        title=f"[bold]{title}[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold bright_cyan",
        title_style="bold white",
        padding=(0, 1),
    )
    for h in headers:
        table.add_column(h, justify="right" if h != headers[0] else "left")

    # Find best row if needed
    best_row = None
    if highlight_best is not None and rows:
        try:
            numeric_vals = []
            for r in rows:
                v = r[highlight_best]
                numeric_vals.append(float(v) if isinstance(v, (int, float)) else float(v))
            best_row = numeric_vals.index(max(numeric_vals))
        except (ValueError, IndexError):
            best_row = None

    for i, row in enumerate(rows):
        style = "bold green" if i == best_row else None
        str_row = []
        for j, cell in enumerate(row):
            if isinstance(cell, float):
                str_row.append(f"{cell:.4f}")
            else:
                str_row.append(str(cell))
        table.add_row(*str_row, style=style)

    console.print()
    console.print(table, justify="center")

    # Also log to file
    _logger.info(f"Table: {title}")
    _logger.info(" | ".join(headers))
    for row in rows:
        _logger.info(" | ".join(f"{c:.4f}" if isinstance(c, float) else str(c) for c in row))


def champion(model_name: str, metrics_dict: dict):
    """Display the champion model with a gold panel."""
    lines = [f"[bold white]{model_name}[/bold white]\n"]
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            lines.append(f"  [cyan]{k}:[/cyan] [bold]{v:.4f}[/bold]")
        else:
            lines.append(f"  [cyan]{k}:[/cyan] [bold]{v}[/bold]")
    content = "\n".join(lines)
    panel = Panel(
        content,
        title="🏆 [bold yellow]BEST MODEL[/bold yellow] 🏆",
        border_style="yellow",
        box=box.DOUBLE_EDGE,
        padding=(1, 3),
    )
    console.print()
    console.print(panel, justify="center")
    _logger.info(f"BEST MODEL: {model_name} — {metrics_dict}")


def saved(path: str):
    """Notify that a file was saved."""
    short = os.path.relpath(path, BASE_DIR) if path.startswith(BASE_DIR) else path
    console.print(f"    [dim]💾 {short}[/dim]")
    _logger.debug(f"Saved: {short}")


def progress_bar(total: int, description: str = "Processing"):
    """Return a Rich Progress context manager.
    
    Usage:
        with progress_bar(12, "EDA Plots") as pb:
            task = pb.add_task("Generating...", total=12)
            for i in range(12):
                ...
                pb.update(task, advance=1, description=f"Plot {i+1}/12")
    """
    return Progress(
        SpinnerColumn(style="bright_cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="cyan", complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def pipeline_complete():
    """Display the final pipeline completion banner."""
    content = (
        "[bold green]All steps completed successfully![/bold green]\n\n"
        "  📊  EDA plots        → [cyan]plots/[/cyan]\n"
        "  🤖  Trained models   → [cyan]models/[/cyan]\n"
        "  📈  MLflow logs      → [cyan]mlruns/[/cyan]\n"
        "  📝  Execution log    → [cyan]logs/[/cyan]\n"
        "  🌐  Start API        → [dim]python api/app.py[/dim]\n"
        "  🐳  Docker           → [dim]docker-compose up --build[/dim]"
    )
    panel = Panel(
        content,
        title="✅ [bold green]PIPELINE COMPLETE[/bold green] ✅",
        border_style="green",
        box=box.DOUBLE_EDGE,
        padding=(1, 3),
    )
    console.print()
    console.print(panel, justify="center")
    console.print()
    elapsed = _logger.handlers[0].stream.name if hasattr(_logger.handlers[0], 'stream') else _log_file
    console.print(f"  [dim]Full log: {os.path.relpath(_log_file, BASE_DIR)}[/dim]")
    _logger.info("Pipeline completed successfully")


def data_summary(shape, n_missing: int, n_duplicates: int, n_features_eng: int = 0):
    """Display a data summary panel."""
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, padding=(0, 2))
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold white")
    table.add_row("Rows", str(shape[0]))
    table.add_row("Columns", str(shape[1]))
    table.add_row("Missing values", str(n_missing))
    table.add_row("Duplicates", str(n_duplicates))
    if n_features_eng > 0:
        table.add_row("Engineered features", str(n_features_eng))
    console.print()
    console.print(table)
    _logger.info(f"Data: {shape[0]} rows, {shape[1]} cols, {n_missing} missing, {n_duplicates} dupes")
