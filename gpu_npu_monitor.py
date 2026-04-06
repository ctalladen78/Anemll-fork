#!/usr/bin/env python3
"""
Real-time GPU/NPU Dashboard for Apple Silicon
Monitors GPU power, ANE (Neural Engine) power, and per-process GPU usage.

# Usage:
# Run with sudo (required for powermetrics)
# sudo python3 gpu_monitor.py

"""

import subprocess
import threading
import time
import re
import sys
from datetime import datetime
from collections import deque

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better display...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "--quiet"])
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live


class GPUMonitor:
    def __init__(self, sample_interval=1.0):
        self.sample_interval = sample_interval
        self.history_length = 60  # Keep 60 samples
        self.gpu_power_history = deque(maxlen=self.history_length)
        self.ane_power_history = deque(maxlen=self.history_length)
        self.process_gpu = {}
        self.current_gpu_power = 0.0
        self.current_ane_power = 0.0
        self.running = False
        self.thread = None
        self.console = Console()

    def parse_powermetrics(self, output):
        """Parse powermetrics output for GPU and ANE metrics."""
        gpu_power = 0.0
        ane_power = 0.0
        processes = {}

        # Parse GPU power
        gpu_match = re.search(r'GPU Power:\s+([\d.]+)\s*mW', output)
        if gpu_match:
            gpu_power = float(gpu_match.group(1)) / 1000.0  # Convert to watts

        # Parse ANE power
        ane_match = re.search(r'ANE Power:\s+([\d.]+)\s*mW', output)
        if ane_match:
            ane_power = float(ane_match.group(1)) / 1000.0

        # Parse per-process GPU time
        proc_pattern = re.findall(r'(\S+.*?)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*?', output)
        for proc in proc_pattern:
            if len(proc) >= 3:
                name = proc[0].strip()
                try:
                    gpu_time = float(proc[2])
                    if gpu_time > 0:
                        processes[name] = gpu_time
                except (ValueError, IndexError):
                    pass

        return gpu_power, ane_power, processes

    def poll_metrics(self):
        """Poll powermetrics for current metrics."""
        try:
            result = subprocess.run([
                'sudo', '/usr/bin/powermetrics',
                '-i', str(int(self.sample_interval * 1000)),
                '--sample', 'gpu_power',
                '-s', 'gpu_power',
                '--show-process-gpu'
            ], capture_output=True, text=True, timeout=self.sample_interval * 2)

            if result.returncode == 0:
                gpu_power, ane_power, processes = self.parse_powermetrics(result.stdout)
                self.current_gpu_power = gpu_power
                self.current_ane_power = ane_power
                self.gpu_power_history.append(gpu_power)
                self.ane_power_history.append(ane_power)
                self.process_gpu = processes

        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"Error polling metrics: {e}")

    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            self.poll_metrics()

    def get_average_gpu(self, last_n=None):
        """Get average GPU power over last N samples."""
        if last_n is None:
            last_n = len(self.gpu_power_history)
        if not self.gpu_power_history:
            return 0.0
        history = list(self.gpu_power_history)[-last_n:]
        return sum(history) / len(history) if history else 0.0

    def get_average_ane(self, last_n=None):
        """Get average ANE power over last N samples."""
        if last_n is None:
            last_n = len(self.ane_power_history)
        if not self.ane_power_history:
            return 0.0
        history = list(self.ane_power_history)[-last_n:]
        return sum(history) / len(history) if history else 0.0


def create_dashboard(monitor):
    """Create the dashboard display."""
    console = Console()
    
    # Header
    console.print("\n[bold cyan]🍎 Apple Silicon GPU/NPU Dashboard[/bold cyan]")
    console.print(f"[dim]Updated: {datetime.now().strftime('%H:%M:%S')}[/dim]\n")
    
    # Current metrics table
    table = Table(title="Current Power Usage", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Current (W)", style="green", justify="right")
    table.add_column("Avg (W)", style="yellow", justify="right")
    table.add_column("Status", style="white")
    
    gpu_current = monitor.current_gpu_power
    gpu_avg = monitor.get_average_gpu()
    ane_current = monitor.current_ane_power
    ane_avg = monitor.get_average_ane()
    
    # Status indicators
    gpu_status = "🔥 Active" if gpu_current > 0.5 else "💤 Idle"
    ane_status = "🧠 Active" if ane_current > 0.1 else "💤 Idle"
    
    table.add_row(
        "GPU (Metal)",
        f"{gpu_current:.2f}",
        f"{gpu_avg:.2f}",
        gpu_status
    )
    table.add_row(
        "ANE (NPU)",
        f"{ane_current:.2f}",
        f"{ane_avg:.2f}",
        ane_status
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{gpu_current + ane_current:.2f}[/bold]",
        f"[bold]{gpu_avg + ane_avg:.2f}[/bold]",
        ""
    )
    console.print(table)
    
    # History bar chart
    console.print("\n[bold]Power History (last 60 seconds)[/bold]")
    
    if monitor.gpu_power_history:
        console.print("[cyan]GPU:  [green]" + "█" * int(gpu_avg * 10) + f" {gpu_avg:.1f}W[/green]")
        console.print("[magenta]ANE:  [green]" + "█" * int(ane_avg * 10) + f" {ane_avg:.1f}W[/green]")
    
    # Top GPU processes
    if monitor.process_gpu:
        console.print("\n[bold]Top GPU Processes[/bold]")
        proc_table = Table(show_header=True)
        proc_table.add_column("Process", style="cyan")
        proc_table.add_column("GPU %", style="green", justify="right")
        
        sorted_procs = sorted(monitor.process_gpu.items(), key=lambda x: x[1], reverse=True)[:5]
        for proc, gpu_pct in sorted_procs:
            proc_table.add_row(proc[:40], f"{gpu_pct:.1f}%")
        
        console.print(proc_table)
    
    console.print("\n[dim]Press Ctrl+C to exit[/dim]")


def main():
    monitor = GPUMonitor(sample_interval=1.0)
    
    console = Console()
    
    # Check for sudo
    result = subprocess.run(['sudo', '-n', 'true'], capture_output=True)
    if result.returncode != 0:
        console.print("[yellow]Note: This script requires sudo for powermetrics.[/yellow]")
        console.print("[dim]Run with: sudo python3 gpu_monitor.py[/dim]")
        console.print("[dim]Or add to sudoers: ctalladen ALL=(ALL) NOPASSWD: /usr/bin/powermetrics[/dim]\n")
    
    try:
        console.print("[yellow]Starting GPU/NPU monitor...[/yellow]")
        monitor.start()
        
        while True:
            create_dashboard(monitor)
            time.sleep(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitor...[/yellow]")
        monitor.stop()
        console.print("[green]Done![/green]")


if __name__ == "__main__":
    main()