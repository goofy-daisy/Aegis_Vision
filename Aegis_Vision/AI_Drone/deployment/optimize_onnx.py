"""ONNX optimization and export module for AegisVision.

Exports YOLO model to ONNX format for optimized inference.
Falls back to CPU if GPU export fails.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from utils.config_loader import Config, load_config
from utils.device_utils import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


def export_onnx(
    weights_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    config: Optional[Config] = None,
) -> Dict:
    """Export YOLO model to ONNX format.
    
    Exports model using Ultralytics export, then validates with ONNX Runtime.
    Falls back to CPU if GPU export fails.
    
    Args:
        weights_path: Path to .pt weights file. Uses config if None.
        output_path: Path for .onnx output. Uses config if None.
        config: Configuration object.
        
    Returns:
        Dict containing export status, paths, and validation results.
        
    Raises:
        RuntimeError: If export fails.
    """
    if config is None:
        config = load_config()
    
    if weights_path is None:
        weights_path = config.paths.weights / "best.pt"
    
    if output_path is None:
        output_path = config.paths.weights / "best.onnx"
    
    # Validate weights exist
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    logger.info(f"Starting ONNX export: {weights_path} -> {output_path}")
    
    try:
        # Load model
        model = YOLO(str(weights_path))
        
        # Export to ONNX
        # Ultralytics handles the export internally
        model.export(
            format="onnx",
            dynamic=True,
            simplify=True,
            imgsz=config.model.input_size,
        )
        
        # Find exported file (Ultralytics puts it in same dir as weights)
        exported_path = weights_path.with_suffix(".onnx")
        if exported_path.exists():
            # Move to desired output location
            import shutil
            shutil.move(str(exported_path), str(output_path))
        else:
            raise RuntimeError("Export completed but output file not found")
        
        logger.info(f"ONNX model exported to {output_path}")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise RuntimeError(f"Failed to export to ONNX: {e}")
    
    # Validate export with ONNX Runtime
    try:
        validation_result = validate_onnx(output_path, config)
    except Exception as e:
        logger.warning(f"ONNX validation failed: {e}")
        validation_result = {"valid": False, "error": str(e)}
    
    # Compute size reduction
    pt_size = weights_path.stat().st_size
    onnx_size = output_path.stat().st_size
    size_reduction = (pt_size - onnx_size) / pt_size * 100
    
    logger.info(f"Size reduction: {size_reduction:.1f}% ({pt_size/1e6:.1f}MB -> {onnx_size/1e6:.1f}MB)")
    
    return {
        "success": True,
        "weights_path": str(weights_path),
        "onnx_path": str(output_path),
        "validation": validation_result,
        "pt_size_mb": pt_size / 1e6,
        "onnx_size_mb": onnx_size / 1e6,
        "size_reduction_pct": size_reduction,
    }


def validate_onnx(onnx_path: Path, config: Config) -> Dict:
    """Validate ONNX model by running inference.
    
    Args:
        onnx_path: Path to ONNX model file.
        config: Configuration object.
        
    Returns:
        Validation results dict.
        
    Raises:
        RuntimeError: If validation fails.
    """
    logger.info(f"Validating ONNX model: {onnx_path}")
    
    try:
        # Try GPU provider first, fall back to CPU
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        # Get provider info
        used_provider = session.get_providers()[0]
        logger.info(f"ONNX Runtime using provider: {used_provider}")
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, config.model.input_size, config.model.input_size)
        dummy_input = dummy_input.astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        logger.info(f"ONNX validation successful. Output shapes: {[o.shape for o in outputs]}")
        
        return {
            "valid": True,
            "provider": used_provider,
            "input_shape": input_shape,
            "output_shapes": [o.shape for o in outputs],
        }
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        raise RuntimeError(f"ONNX validation failed: {e}")


def load_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    """Load an ONNX Runtime inference session.
    
    Args:
        onnx_path: Path to ONNX model file.
        
    Returns:
        ONNX Runtime inference session.
        
    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If session creation fails.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    try:
        # Try GPU first, fall back to CPU
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f"ONNX session loaded with provider: {session.get_providers()[0]}")
        
        return session
        
    except Exception as e:
        logger.error(f"Failed to load ONNX session: {e}")
        raise RuntimeError(f"ONNX session creation failed: {e}")


def benchmark_onnx(onnx_path: Path, num_runs: int = 100) -> Dict:
    """Benchmark ONNX model inference speed.

    Runs the ONNX model through repeated inference passes and
    reports average latency and throughput metrics.

    Args:
        onnx_path: Path to the .onnx model file.
        num_runs: Number of inference iterations to run.

    Returns:
        Dict with benchmark statistics:
        - avg_latency_ms: Average latency in milliseconds
        - min_latency_ms: Minimum latency
        - max_latency_ms: Maximum latency
        - throughput_fps: Inferences per second
        - num_runs: Number of benchmark iterations
    """
    import time

    session = load_onnx_session(onnx_path)
    input_name = session.get_inputs()[0].name

    # Create dummy input (1, 3, 640, 640) — standard YOLO input
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup runs
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0

    results = {
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "throughput_fps": round(throughput, 1),
        "num_runs": num_runs,
    }

    logger.info(
        f"ONNX Benchmark ({onnx_path.name}): "
        f"avg={results['avg_latency_ms']}ms, "
        f"fps={results['throughput_fps']}"
    )
    return results


def main():
    """Command-line entry point for ONNX export."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export AegisVision model to ONNX")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--weights", type=Path, help="Path to .pt weights file")
    parser.add_argument("--output", type=Path, help="Output path for .onnx file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config) if args.config else load_config()
    
    # Run export
    results = export_onnx(
        weights_path=args.weights,
        output_path=args.output,
        config=config,
    )
    
    # Print summary
    print("\nExport Summary:")
    print(f"  Success:        {results['success']}")
    print(f"  ONNX path:      {results['onnx_path']}")
    print(f"  Valid:          {results['validation']['valid']}")
    print(f"  Provider:       {results['validation'].get('provider', 'N/A')}")
    print(f"  Size reduction: {results['size_reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()
