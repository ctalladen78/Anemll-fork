import os
import time
import psutil
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.align import Align
from rich.text import Text
from datetime import datetime

# Pathing
LOG_FILE = "benchmark.log"

def get_system_stats():
    """Fetches real-time macOS RAM pressure."""
    mem = psutil.virtual_memory()
    percent = mem.percent
    gb_used = mem.used / (1024 ** 3)
    gb_total = mem.total / (1024 ** 3)
    
    color = "green" if percent < 75 else "yellow" if percent < 90 else "red"
    return f"[{color}]{percent:.1f}% ({gb_used:.1f}GB / {gb_total:.1f}GB)[/{color}]"

def get_latest_log_lines(n=15):
    """Reads the last n lines from benchmark.log."""
    if not os.path.exists(LOG_FILE):
        return "[dim]Waiting for benchmark output...[/dim]"
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            return "".join(lines[-n:]).strip()
    except Exception as e:
        return f"[red]Error reading logs: {str(e)}[/red]"

def build_layout():
    """Constructs the Rich TUI layout."""
    sys_ram = get_system_stats()
    log_content = get_latest_log_lines()
    now = datetime.now().strftime("%H:%M:%S")

    # --- Header ---
    mem_color = "red bold" if "red" in sys_ram else "green"
    header_text = Text(f"🚀 Llama-CPP TurboQuant Dashboard | {now} | RAM Usage: ", style="bold")
    header_text.append_text(Text.from_markup(sys_ram))
    
    # --- Main Panel (Log Stream) ---
    main_panel = Panel(
        Text.from_markup(log_content),
        title="[b]Benchmark Mind Stream (benchmark.log)[/b]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # --- Stats Panel (Placeholder for now) ---
    stats_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    stats_table.add_column("Metric")
    stats_table.add_column("Value")
    stats_table.add_row("Active Model", "[bold yellow]Qwen3.5-35B-A3B[/bold yellow]")
    stats_table.add_row("Quant Detail", "TurboQuant (TQ4_1S)")
    stats_table.add_row("KV Cache", "Asymmetric (K=Q8, V=TQ4)")
    
    stats_panel = Panel(stats_table, title="[b]Configuration[/b]", border_style="magenta")

    # --- Layout ---
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body")
    )
    layout["body"].split_row(
        Layout(name="main", ratio=3),
        Layout(name="side", ratio=1)
    )
    
    layout["header"].update(Panel(Align.center(header_text), border_style="blue"))
    layout["main"].update(main_panel)
    layout["side"].update(stats_panel)
    
    return layout

if __name__ == "__main__":
    try:
        with Live(build_layout(), refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(0.5)
                live.update(build_layout())
    except KeyboardInterrupt:
        print("\nDashboard exited.")
