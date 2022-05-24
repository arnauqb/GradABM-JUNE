from pathlib import Path
import os

torch_june_path = Path(os.path.abspath(__file__)).parent.parent

default_config_path = (
    Path(os.path.abspath(__file__)).parent.parent / "configs/default.yaml"
)
