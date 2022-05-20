from pathlib import Path
import os

default_config_path = (
    Path(os.path.abspath(__file__)).parent.parent / "configs/default.yaml"
)
