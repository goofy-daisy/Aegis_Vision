"""Configuration loader for AegisVision.

Loads and validates config.yaml, exposing a typed configuration object
that can be imported by any module. All paths are resolved to absolute
pathlib.Path objects relative to the project root.
"""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PathsConfig:
    """Paths configuration section."""
    data_root: Path
    visdrone_raw: Path
    dota_raw: Path
    processed: Path
    weights: Path
    logs: Path
    mlflow_tracking: Path
    threat_model: Path


@dataclass
class ModelConfig:
    """Model configuration section."""
    yolo_weights_pretrained: str
    yolo_model_size: str
    input_size: int
    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int
    device: str


@dataclass
class TrainingConfig:
    """Training configuration section."""
    epochs: int
    batch_size: int
    learning_rate: float
    val_split: float
    resume: bool
    patience: int
    save_period: int


@dataclass
class TrackingConfig:
    """Tracking configuration section."""
    max_age: int
    min_hits: int
    iou_threshold: float
    max_tracks: int


@dataclass
class ThreatConfig:
    """Threat scoring configuration section."""
    weights: Dict[str, float]
    dangerous_classes: List[str]
    high_threat_threshold: float
    medium_threat_threshold: float
    proximity_max_meters: float
    speed_max_mps: float


@dataclass
class SimulationConfig:
    """Simulation configuration section."""
    fog_intensity: float
    blur_kernel_max: int
    noise_std: float
    telemetry_update_hz: int


@dataclass
class ApiConfig:
    """API configuration section."""
    host: str
    port: int
    max_upload_mb: int
    websocket_fps: int


@dataclass
class DashboardConfig:
    """Dashboard configuration section."""
    port: int
    max_fps_display: int
    map_center_lat: float
    map_center_lon: float


@dataclass
class MlflowConfig:
    """MLflow configuration section."""
    experiment_name: str
    run_name_prefix: str
    tags: Dict[str, str]


@dataclass
class Config:
    """Master configuration object containing all sections."""
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    threat: ThreatConfig
    simulation: SimulationConfig
    api: ApiConfig
    dashboard: DashboardConfig
    mlflow: MlflowConfig


# Singleton instance
_config_instance: Optional[Config] = None


def _resolve_path(path_str: str, project_root: Path) -> Path:
    """Resolve a path string to an absolute Path object.
    
    Args:
        path_str: The path string from config, may be relative or absolute.
        project_root: The project root directory for resolving relative paths.
        
    Returns:
        An absolute Path object.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validate_paths(config_data: Dict[str, Any], project_root: Path) -> None:
    """Validate that all required path keys exist in the config.
    
    Args:
        config_data: The raw config dictionary from YAML.
        project_root: The project root directory.
        
    Raises:
        ValueError: If required path keys are missing.
    """
    required_path_keys = [
        "data_root", "visdrone_raw", "dota_raw", "processed", 
        "weights", "logs", "mlflow_tracking", "threat_model"
    ]
    
    paths_data = config_data.get("paths", {})
    missing_keys = [key for key in required_path_keys if key not in paths_data]
    
    if missing_keys:
        raise ValueError(f"Missing required path keys in config: {missing_keys}")
    
    logger.debug(f"Validated {len(required_path_keys)} required path keys")


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load and validate the configuration from config.yaml.
    
    This function returns a singleton Config object. Subsequent calls
    return the same instance unless the config file is reloaded.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        
    Returns:
        A validated Config dataclass instance with all settings.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config validation fails.
    """
    global _config_instance
    
    # Return existing singleton if already loaded
    if _config_instance is not None:
        return _config_instance
    
    # Determine config file path
    if config_path is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    
    # Validate config file exists
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config YAML: {e}")
        raise ValueError(f"Invalid YAML in config file: {e}")
    
    # Determine project root for path resolution
    project_root = config_path.parent.parent
    
    # Validate required paths exist
    _validate_paths(config_data, project_root)
    
    # Resolve all paths to absolute Path objects
    paths_data = config_data["paths"]
    paths_config = PathsConfig(
        data_root=_resolve_path(paths_data["data_root"], project_root),
        visdrone_raw=_resolve_path(paths_data["visdrone_raw"], project_root),
        dota_raw=_resolve_path(paths_data["dota_raw"], project_root),
        processed=_resolve_path(paths_data["processed"], project_root),
        weights=_resolve_path(paths_data["weights"], project_root),
        logs=_resolve_path(paths_data["logs"], project_root),
        mlflow_tracking=_resolve_path(paths_data["mlflow_tracking"], project_root),
        threat_model=_resolve_path(paths_data["threat_model"], project_root),
    )
    
    # Create typed config sections
    model_data = config_data["model"]
    model_config = ModelConfig(
        yolo_weights_pretrained=model_data["yolo_weights_pretrained"],
        yolo_model_size=model_data["yolo_model_size"],
        input_size=model_data["input_size"],
        confidence_threshold=model_data["confidence_threshold"],
        nms_iou_threshold=model_data["nms_iou_threshold"],
        max_detections=model_data["max_detections"],
        device=model_data["device"],
    )
    
    training_data = config_data["training"]
    training_config = TrainingConfig(
        epochs=training_data["epochs"],
        batch_size=training_data["batch_size"],
        learning_rate=training_data["learning_rate"],
        val_split=training_data["val_split"],
        resume=training_data["resume"],
        patience=training_data["patience"],
        save_period=training_data["save_period"],
    )
    
    tracking_data = config_data["tracking"]
    tracking_config = TrackingConfig(
        max_age=tracking_data["max_age"],
        min_hits=tracking_data["min_hits"],
        iou_threshold=tracking_data["iou_threshold"],
        max_tracks=tracking_data["max_tracks"],
    )
    
    threat_data = config_data["threat"]
    threat_config = ThreatConfig(
        weights=threat_data["weights"],
        dangerous_classes=threat_data["dangerous_classes"],
        high_threat_threshold=threat_data["high_threat_threshold"],
        medium_threat_threshold=threat_data["medium_threat_threshold"],
        proximity_max_meters=threat_data["proximity_max_meters"],
        speed_max_mps=threat_data["speed_max_mps"],
    )
    
    simulation_data = config_data["simulation"]
    simulation_config = SimulationConfig(
        fog_intensity=simulation_data["fog_intensity"],
        blur_kernel_max=simulation_data["blur_kernel_max"],
        noise_std=simulation_data["noise_std"],
        telemetry_update_hz=simulation_data["telemetry_update_hz"],
    )
    
    api_data = config_data["api"]
    api_config = ApiConfig(
        host=api_data["host"],
        port=api_data["port"],
        max_upload_mb=api_data["max_upload_mb"],
        websocket_fps=api_data["websocket_fps"],
    )
    
    dashboard_data = config_data["dashboard"]
    dashboard_config = DashboardConfig(
        port=dashboard_data["port"],
        max_fps_display=dashboard_data["max_fps_display"],
        map_center_lat=dashboard_data["map_center_lat"],
        map_center_lon=dashboard_data["map_center_lon"],
    )
    
    mlflow_data = config_data["mlflow"]
    mlflow_config = MlflowConfig(
        experiment_name=mlflow_data["experiment_name"],
        run_name_prefix=mlflow_data["run_name_prefix"],
        tags=mlflow_data["tags"],
    )
    
    # Create master config object
    _config_instance = Config(
        paths=paths_config,
        model=model_config,
        training=training_config,
        tracking=tracking_config,
        threat=threat_config,
        simulation=simulation_config,
        api=api_config,
        dashboard=dashboard_config,
        mlflow=mlflow_config,
    )
    
    logger.info(f"Configuration loaded successfully from {config_path}")
    return _config_instance


def reload_config(config_path: Optional[Path] = None) -> Config:
    """Force reload the configuration from disk.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        
    Returns:
        A fresh Config instance loaded from disk.
    """
    global _config_instance
    _config_instance = None
    return load_config(config_path)
