from pathlib import Path
import yaml


def find_project_root(config_name: str = "preprocessing.yaml") -> Path:
    cwd = Path.cwd().resolve()

    for path in [cwd, *cwd.parents]:
        if (path / "configs" / config_name).exists():
            return path

    raise FileNotFoundError(f"Could not find configs/{config_name}.")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)