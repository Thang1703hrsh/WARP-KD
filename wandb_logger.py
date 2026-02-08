"""
Minimal wandb logger for contra-kd training.
"""
import os
from typing import Dict, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_wandb_config_from_yaml(base_path: str = ".") -> Optional[Dict[str, Any]]:
    """Load wandb configuration from YAML file."""
    import yaml
    config_path = os.path.join(base_path, "wandb_config.yaml")
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('wandb', {})
    except Exception as e:
        print(f"Warning: Failed to load wandb config from {config_path}: {e}")
        return None


class WandbLogger:
    """Minimal wandb logger."""
    
    def __init__(self):
        self.enabled = False
        self.run = None
    
    def init(self, project: str, name: str, config: Dict[str, Any], wandb_key: Optional[str] = None, base_path: str = ".") -> bool:
        """Initialize wandb logging. If wandb_key is None, will try to load from YAML."""
        if not WANDB_AVAILABLE:
            return False
        
        # If no key provided, try loading from YAML
        if not wandb_key:
            yaml_config = load_wandb_config_from_yaml(base_path)
            if yaml_config and yaml_config.get('enabled', False):
                wandb_key = yaml_config.get('key')
                if not project:
                    project = yaml_config.get('project', 'contra-kd')
        
        if not wandb_key:
            return False
        
        try:
            # Auto-login if key provided
            os.environ['WANDB_API_KEY'] = wandb_key
            wandb.login(key=wandb_key)
            
            self.run = wandb.init(project=project, name=name, config=config, reinit=True)
            self.enabled = True
            print(f"✓ Wandb initialized successfully (project: {project})")
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            return False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.enabled and self.run:
            self.run.log(metrics, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run:
            self.run.finish()


# Global logger instance
_logger = WandbLogger()


def init_wandb(project: str, name: str, config: Dict[str, Any], wandb_key: Optional[str] = None, base_path: str = ".") -> bool:
    """Initialize the global wandb logger."""
    return _logger.init(project, name, config, wandb_key, base_path)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics using the global logger."""
    _logger.log(metrics, step)


def finish_wandb():
    """Finish wandb run."""
    _logger.finish()
