#!/usr/bin/env python3
"""
Unified launcher for all production services.

Starts:
1. vLLM service for Qwen2.5-VL
2. Ray Serve deployment for Qwen-Image-Edit-2509
3. Gradio frontend

All services are configured via production_pipeline.yaml.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ServiceManager:
    """Manages lifecycle of all production services."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.processes: List[subprocess.Popen] = []
        self.log_dir = project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def start_vllm_service(self) -> Optional[subprocess.Popen]:
        """Start vLLM service for Qwen2.5-VL."""
        logger.info("Starting vLLM service...")
        
        vl_config = self.config['models']['qwen_vl']
        gpu_alloc = self.config['gpu_allocation']['vl_service']
        endpoint = self.config['services']['vl_endpoint']
        port = endpoint.split(':')[-1].split('/')[0]
        
        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_alloc)
        
        # Build command
        cmd = [
            'vllm', 'serve', vl_config['model_name'],
            '--tensor-parallel-size', str(vl_config['tensor_parallel_size']),
            '--max-model-len', str(vl_config['max_model_len']),
            '--gpu-memory-utilization', str(vl_config['gpu_memory_utilization']),
            '--port', str(port),
            '--host', '0.0.0.0',
            '--trust-remote-code',
        ]
        
        logger.info(f"  Model: {vl_config['model_name']}")
        logger.info(f"  GPUs: {gpu_alloc}")
        logger.info(f"  Port: {port}")
        
        # Start process
        log_file = self.log_dir / "vllm_service.log"
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root
            )
        
        logger.info(f"  PID: {process.pid}")
        logger.info(f"  Log: {log_file}")
        
        self.processes.append(process)
        return process
    
    def start_ray_serve(self) -> Optional[subprocess.Popen]:
        """Start Ray Serve deployment for Qwen-Image-Edit-Plus."""
        logger.info("Starting Ray Serve deployment...")
        
        model_config = self.config['models']['qwen_image_edit']
        gpu_alloc = self.config['gpu_allocation']['image_edit_service']
        
        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_alloc)
        
        # Build command
        cmd = [
            sys.executable,
            str(project_root / 'script' / 'launch_ray_serve.py'),
            '--config', self.config_path,
            '--detached',
        ]
        
        logger.info(f"  Model: {model_config['model_name']}")
        logger.info(f"  GPUs available: {gpu_alloc}")
        logger.info(f"  Replicas: {model_config.get('num_replicas', 6)}")
        logger.info(f"  GPUs per replica: {model_config.get('gpus_per_replica', 1)}")
        logger.info(f"  Total GPUs needed: {model_config.get('num_replicas', 6) * model_config.get('gpus_per_replica', 1)}")
        if model_config.get('lora_path'):
            logger.info(f"  LoRA: {model_config['lora_path']}")
        
        # Start process
        log_file = self.log_dir / "ray_serve.log"
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root
            )
        
        logger.info(f"  PID: {process.pid}")
        logger.info(f"  Log: {log_file}")
        
        self.processes.append(process)
        return process
    
    def start_frontend(self) -> Optional[subprocess.Popen]:
        """Start Gradio frontend."""
        logger.info("Starting Gradio frontend...")
        
        frontend_config = self.config['frontend']
        
        # Build command
        cmd = [
            sys.executable,
            str(project_root / 'script' / 'gradio_production_frontend.py'),
            '--config', self.config_path,
        ]
        
        logger.info(f"  Host: {frontend_config['host']}")
        logger.info(f"  Port: {frontend_config['port']}")
        
        # Start process
        log_file = self.log_dir / "frontend.log"
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root
            )
        
        logger.info(f"  PID: {process.pid}")
        logger.info(f"  Log: {log_file}")
        
        self.processes.append(process)
        return process
    
    def check_health(self, service: str, url: str, timeout: int = 60) -> bool:
        """Check if a service is healthy."""
        import requests
        
        logger.info(f"Waiting for {service} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"  {service} is ready!")
                    return True
            except Exception:
                pass
            
            time.sleep(2)
        
        logger.warning(f"  {service} health check timed out")
        return False
    
    def get_gradio_share_link(self, timeout: int = 30) -> Optional[str]:
        """Extract Gradio share link from frontend log."""
        log_file = self.log_dir / "frontend.log"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        content = f.read()
                        # Look for explicit marker first
                        marker_pattern = r'GRADIO_SHARE_LINK:\s*(https?://[^\s]+)'
                        match = re.search(marker_pattern, content)
                        if match:
                            return match.group(1)
                        
                        # Fallback: Look for gradio.live or similar share URL patterns
                        pattern = r'(https?://[a-zA-Z0-9\-]+\.gradio\.live)'
                        match = re.search(pattern, content)
                        if match:
                            return match.group(1)
            except Exception:
                pass
            time.sleep(1)
        
        return None
    
    def start_all(self, skip_vllm: bool = False, skip_ray: bool = False, skip_frontend: bool = False):
        """Start all services."""
        logger.info("=" * 60)
        logger.info("  Starting Production Services")
        logger.info("=" * 60)
        logger.info(f"Config: {self.config_path}")
        logger.info("")
        
        # Start services in order
        if not skip_vllm:
            self.start_vllm_service()
            time.sleep(5)  # Give it time to initialize
        
        if not skip_ray:
            self.start_ray_serve()
            time.sleep(10)  # Give it time to initialize
        
        if not skip_frontend:
            self.start_frontend()
            time.sleep(3)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("  All Services Started!")
        logger.info("=" * 60)
        
        if not skip_vllm:
            vl_endpoint = self.config['services']['vl_endpoint']
            logger.info(f"  vLLM: {vl_endpoint}")
        
        if not skip_ray:
            image_endpoint = self.config['services']['image_edit_endpoint']
            logger.info(f"  Ray Serve: {image_endpoint}/edit")
        
        if not skip_frontend:
            frontend = self.config['frontend']
            logger.info(f"  Frontend (Local): http://{frontend['host']}:{frontend['port']}")
            
            # If share is enabled, try to extract the public link
            if frontend.get('share', False):
                logger.info("  Extracting Gradio share link...")
                share_link = self.get_gradio_share_link(timeout=30)
                if share_link:
                    logger.info(f"  Frontend (Public): {share_link}")
                else:
                    logger.info("  Frontend (Public): Check logs/frontend.log for share link")
        
        logger.info("=" * 60)
        logger.info(f"  Logs: {self.log_dir}/")
        logger.info("=" * 60)
    
    def stop_all(self):
        """Stop all services."""
        logger.info("Stopping all services...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        
        logger.info("All services stopped")
    
    def wait(self):
        """Wait for user interrupt."""
        try:
            logger.info("Press Ctrl+C to stop all services")
            while True:
                time.sleep(1)
                # Check if any process died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        logger.error(f"Service {i} (PID {process.pid}) died unexpectedly!")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")


def main():
    parser = argparse.ArgumentParser(
        description="Launch all production services"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_pipeline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip starting vLLM service"
    )
    parser.add_argument(
        "--skip-ray",
        action="store_true",
        help="Skip starting Ray Serve"
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip starting frontend"
    )
    
    args = parser.parse_args()
    
    manager = ServiceManager(args.config)
    
    try:
        manager.start_all(
            skip_vllm=args.skip_vllm,
            skip_ray=args.skip_ray,
            skip_frontend=args.skip_frontend
        )
        manager.wait()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()

