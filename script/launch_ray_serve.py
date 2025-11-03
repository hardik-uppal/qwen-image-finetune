#!/usr/bin/env python3
"""
Launch Ray Serve deployment for Qwen-Image-Edit-Plus.

Reads configuration from production_pipeline.yaml and deploys the model
across multiple GPUs with automatic load balancing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import ray
from ray import serve

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.serving.image_edit_server import QwenImageEditDeployment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Ray Serve deployment for Qwen-Image-Edit-Plus"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_pipeline.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run in detached mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Get GPU allocation
    gpu_alloc = config['gpu_allocation']['image_edit_service']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_alloc)
    
    logger.info(f"GPU allocation: {gpu_alloc}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Get service configuration
    ray_config = config.get('ray_serve', {})
    http_options = ray_config.get('http_options', {})
    host = http_options.get('host', '0.0.0.0')
    port = http_options.get('port', 8001)
    
    # Get model configuration
    model_config = config['models']['qwen_image_edit']
    num_replicas = model_config.get('num_replicas', 6)
    gpus_per_replica = model_config.get('gpus_per_replica', 1)
    max_concurrent = model_config.get('max_concurrent_queries_per_replica', 2)
    
    logger.info(f"Starting Ray Serve on {host}:{port}")
    logger.info(f"Number of replicas: {num_replicas}")
    logger.info(f"GPUs per replica: {gpus_per_replica}")
    logger.info(f"Total GPUs needed: {num_replicas * gpus_per_replica}")
    logger.info(f"Max concurrent queries per replica: {max_concurrent}")
    
    if model_config.get('lora_path'):
        logger.info(f"LoRA path: {model_config['lora_path']}")
    
    # Initialize Ray with GPU resources FIRST
    if not ray.is_initialized():
        logger.info(f"Initializing Ray with {len(gpu_alloc)} GPUs...")
        ray.init(
            num_gpus=len(gpu_alloc),  # Tell Ray how many GPUs are available
            ignore_reinit_error=True,
        )
        logger.info(f"Ray initialized successfully with {len(gpu_alloc)} GPUs")
    
    # Initialize Ray Serve
    serve.start(
        detached=args.detached,
        http_options={"host": host, "port": port}
    )
    
    # Get autoscaling config
    autoscaling_config = config.get('ray_serve', {}).get('autoscaling_config', {})
    if not autoscaling_config:
        autoscaling_config = None
    
    # Deploy the model
    # Note: Cannot set both num_replicas and autoscaling_config
    deployment_options = {
        "max_ongoing_requests": max_concurrent,
        "ray_actor_options": {"num_gpus": gpus_per_replica},  # Multi-GPU per replica for sharding
    }
    
    if autoscaling_config:
        deployment_options["autoscaling_config"] = autoscaling_config
    else:
        deployment_options["num_replicas"] = num_replicas
    
    deployment = QwenImageEditDeployment.options(**deployment_options)
    
    serve.run(
        deployment.bind(config),
        name="qwen_image_edit",
        route_prefix="/edit"
    )
    
    logger.info("=" * 60)
    logger.info("  Ray Serve Deployment Successful!")
    logger.info("=" * 60)
    logger.info(f"  Endpoint: http://{host}:{port}/edit")
    logger.info(f"  Replicas: {num_replicas}")
    logger.info(f"  GPUs per replica: {gpus_per_replica}")
    logger.info(f"  Model: {model_config['model_name']}")
    if model_config.get('lora_path'):
        logger.info(f"  LoRA: {model_config['lora_path']}")
    logger.info("=" * 60)
    
    # Keep process alive to maintain Ray Serve
    logger.info("Ray Serve is running. Press Ctrl+C to stop.")
    logger.info(f"Detached mode: {args.detached}")
    try:
        import signal
        signal.pause()
    except KeyboardInterrupt:
        logger.info("Shutting down Ray Serve...")
        serve.shutdown()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

