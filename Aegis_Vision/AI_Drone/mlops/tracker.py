"""MLflow experiment tracking module for AegisVision.

Provides MLflow integration for experiment management, metric logging,
and model registry. Tracks training runs and manages model artifacts.
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for AegisVision.
    
    Manages experiment runs, metric logging, parameter tracking, and
    model registration with MLflow.
    
    Attributes:
        cfg: Configuration with MLflow settings.
        client: MLflow client instance.
        experiment_id: Current experiment ID.
        run: Active MLflow run.
        run_start_time: Timestamp for run duration calculation.
    """
    
    def __init__(self, cfg: Config):
        """Initialize MLflow tracker with configuration.
        
        Args:
            cfg: Configuration with MLflow settings.
        """
        self.cfg = cfg
        self.client = MlflowClient()
        self.experiment_id: Optional[str] = None
        self.run: Optional[mlflow.ActiveRun] = None
        self.run_start_time: Optional[float] = None
        
        # Set tracking URI
        from pathlib import Path
        mlflow_path = Path(cfg.paths.mlflow_tracking)
        mlflow_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlflow_path.as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        
        # Set or create experiment
        self._setup_experiment()
        
        logger.info(f"MLflowTracker initialized: experiment={cfg.mlflow.experiment_name}")
    
    def _setup_experiment(self) -> None:
        """Setup MLflow experiment, creating if it doesn't exist."""
        experiment_name = self.cfg.mlflow.experiment_name
        
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=str(self.cfg.paths.mlflow_tracking / "artifacts"),
            )
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.debug(f"Using existing experiment: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional run name. Auto-generated if not provided.
            
        Returns:
            Active MLflow run instance.
            
        Raises:
            RuntimeError: If a run is already active.
        """
        if self.run is not None:
            raise RuntimeError("A run is already active. End it before starting a new one.")
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.cfg.mlflow.run_name_prefix}_{timestamp}"
        
        # Start run
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )
        self.run_start_time = time.time()
        
        # Set tags
        for key, value in self.cfg.mlflow.tags.items():
            mlflow.set_tag(key, value)
        
        # Log config parameters
        self._log_config_params()
        
        logger.info(f"MLflow run started: {run_name} (ID: {self.run.info.run_id})")
        return self.run
    
    def _log_config_params(self) -> None:
        """Log configuration parameters to MLflow."""
        params = {
            "model.yolo_model_size": self.cfg.model.yolo_model_size,
            "model.input_size": self.cfg.model.input_size,
            "model.confidence_threshold": self.cfg.model.confidence_threshold,
            "training.epochs": self.cfg.training.epochs,
            "training.batch_size": self.cfg.training.batch_size,
            "training.learning_rate": self.cfg.training.learning_rate,
            "tracking.max_age": self.cfg.tracking.max_age,
            "tracking.min_hits": self.cfg.tracking.min_hits,
        }
        
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, any]) -> None:
        """Log parameters to the active run.
        
        Args:
            params: Dictionary of parameter names and values.
        """
        if self.run is None:
            logger.warning("No active run to log parameters to")
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the active run.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for the metrics.
        """
        if self.run is None:
            logger.warning("No active run to log metrics to")
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        """Log a file as an artifact.
        
        Args:
            path: Path to the artifact file.
            artifact_path: Optional subdirectory within the artifact directory.
        """
        if self.run is None:
            logger.warning("No active run to log artifact to")
            return
        if not path.exists():
            logger.warning(f"Artifact file not found: {path}")
            return
        try:
            mlflow.log_artifact(str(path), artifact_path)
            logger.debug(f"Logged artifact: {path}")
        except Exception as e:
            logger.warning(f"Artifact logging skipped (Windows path issue): {e}")
    
    def log_model(self, model_path: Path, model_name: str) -> None:
        """Register a model in MLflow Model Registry.
        
        Args:
            model_path: Path to the model file.
            model_name: Name for the registered model.
        """
        if self.run is None:
            logger.warning("No active run to register model from")
            return
        try:
            mlflow.log_artifact(str(model_path), artifact_path="model")
            model_uri = f"runs:/{self.run.info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, model_name)
            logger.info(f"Model registered: {model_name} v{registered_model.version}")
        except Exception as e:
            logger.warning(f"Model registration skipped (Windows path issue): {e}")
    
    def end_run(self) -> None:
        """End the active MLflow run."""
        if self.run is None:
            return
        
        # Log duration
        if self.run_start_time:
            duration = time.time() - self.run_start_time
            mlflow.log_metric("duration_seconds", duration)
        
        run_id = self.run.info.run_id
        mlflow.end_run()
        
        self.run = None
        self.run_start_time = None
        
        logger.info(f"MLflow run ended: {run_id}")
    
    def get_run_info(self) -> Optional[Dict]:
        """Get information about the active run.
        
        Returns:
            Dictionary with run information or None if no active run.
        """
        if self.run is None:
            return None
        
        return {
            "run_id": self.run.info.run_id,
            "run_name": self.run.info.run_name,
            "experiment_id": self.run.info.experiment_id,
            "start_time": self.run.info.start_time,
            "status": self.run.info.status,
        }
