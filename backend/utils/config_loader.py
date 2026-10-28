import os
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Default to root configs/config.yaml
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
