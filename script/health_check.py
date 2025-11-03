#!/usr/bin/env python3
"""
Health check and monitoring script for production services.

Checks:
- vLLM service health
- Ray Serve deployment health
- GPU utilization
- Service response times
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import aiohttp
import yaml
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

console = Console()


class ServiceHealthChecker:
    """Monitor health of all production services."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vl_endpoint = self.config['services']['vl_endpoint']
        self.edit_endpoint = self.config['services']['image_edit_endpoint']
    
    async def check_vllm(self) -> Dict[str, Any]:
        """Check vLLM service health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.vl_endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return {
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "status_code": response.status,
                        "endpoint": self.vl_endpoint,
                    }
        except Exception as e:
            return {
                "status": "down",
                "error": str(e),
                "endpoint": self.vl_endpoint,
            }
    
    async def check_ray_serve(self) -> Dict[str, Any]:
        """Check Ray Serve deployment health."""
        try:
            async with aiohttp.ClientSession() as session:
                # Try to hit the health endpoint
                async with session.get(
                    f"{self.edit_endpoint}/-/healthz",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return {
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "status_code": response.status,
                        "endpoint": self.edit_endpoint,
                    }
        except Exception as e:
            return {
                "status": "down",
                "error": str(e),
                "endpoint": self.edit_endpoint,
            }
    
    def get_gpu_stats(self) -> Optional[Dict[str, Any]]:
        """Get GPU utilization statistics."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpus.append({
                            "id": int(parts[0]),
                            "name": parts[1],
                            "utilization": int(parts[2]),
                            "memory_used": int(parts[3]),
                            "memory_total": int(parts[4]),
                            "temperature": int(parts[5]),
                        })
            
            return {"gpus": gpus}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_full_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        vllm_health, ray_health = await asyncio.gather(
            self.check_vllm(),
            self.check_ray_serve()
        )
        
        gpu_stats = self.get_gpu_stats()
        
        return {
            "vllm": vllm_health,
            "ray_serve": ray_health,
            "gpus": gpu_stats,
        }


def render_status_table(status: Dict[str, Any]) -> Table:
    """Render status as a rich table."""
    table = Table(title="Service Health Status")
    
    table.add_column("Service", style="cyan", width=20)
    table.add_column("Status", width=15)
    table.add_column("Endpoint", style="dim")
    
    # vLLM status
    vllm = status['vllm']
    vllm_status = "✅ Healthy" if vllm['status'] == 'healthy' else f"❌ {vllm['status'].title()}"
    table.add_row(
        "vLLM (Qwen2.5-VL)",
        vllm_status,
        vllm.get('endpoint', 'N/A')
    )
    
    # Ray Serve status
    ray = status['ray_serve']
    ray_status = "✅ Healthy" if ray['status'] == 'healthy' else f"❌ {ray['status'].title()}"
    table.add_row(
        "Ray Serve (Image Edit)",
        ray_status,
        ray.get('endpoint', 'N/A')
    )
    
    return table


def render_gpu_table(gpu_stats: Dict[str, Any]) -> Table:
    """Render GPU statistics as a table."""
    table = Table(title="GPU Status")
    
    table.add_column("ID", justify="right", width=5)
    table.add_column("Name", width=25)
    table.add_column("Util %", justify="right", width=8)
    table.add_column("Memory", justify="right", width=20)
    table.add_column("Temp °C", justify="right", width=8)
    
    if "error" in gpu_stats:
        table.add_row("N/A", "Error", gpu_stats["error"], "", "")
        return table
    
    for gpu in gpu_stats.get("gpus", []):
        util_style = "green" if gpu['utilization'] < 50 else "yellow" if gpu['utilization'] < 80 else "red"
        mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
        mem_style = "green" if mem_pct < 50 else "yellow" if mem_pct < 80 else "red"
        temp_style = "green" if gpu['temperature'] < 70 else "yellow" if gpu['temperature'] < 85 else "red"
        
        table.add_row(
            str(gpu['id']),
            gpu['name'],
            f"[{util_style}]{gpu['utilization']}[/{util_style}]",
            f"[{mem_style}]{gpu['memory_used']}/{gpu['memory_total']} MB[/{mem_style}]",
            f"[{temp_style}]{gpu['temperature']}[/{temp_style}]"
        )
    
    return table


async def monitor_continuous(checker: ServiceHealthChecker, interval: int = 5):
    """Continuously monitor services and display live updates."""
    
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                status = await checker.get_full_status()
                
                # Render tables
                service_table = render_status_table(status)
                gpu_table = render_gpu_table(status['gpus'])
                
                # Create panel
                panel = Panel(
                    f"{service_table}\n\n{gpu_table}",
                    title="Production Pipeline Health Monitor",
                    subtitle=f"Refresh: {interval}s | Press Ctrl+C to exit"
                )
                
                live.update(panel)
                
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                await asyncio.sleep(interval)


async def check_once(checker: ServiceHealthChecker, output_format: str = "table"):
    """Check health once and display results."""
    status = await checker.get_full_status()
    
    if output_format == "json":
        console.print_json(data=status)
    else:
        console.print(render_status_table(status))
        console.print()
        console.print(render_gpu_table(status['gpus']))


def main():
    parser = argparse.ArgumentParser(description="Health check for production services")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_pipeline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuously monitor services"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )
    
    args = parser.parse_args()
    
    checker = ServiceHealthChecker(args.config)
    
    if args.monitor:
        try:
            asyncio.run(monitor_continuous(checker, args.interval))
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")
    else:
        asyncio.run(check_once(checker, args.format))


if __name__ == "__main__":
    main()

