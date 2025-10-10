from __future__ import annotations

from pathlib import Path
import importlib.util
from types import ModuleType
import yaml


def import_module_from_path(module_path: str | Path) -> ModuleType:
    """Dynamically import a Python module from an arbitrary file path.

    This works even if directories contain spaces and aren't valid Python packages.
    """
    module_path = str(module_path)
    spec = importlib.util.spec_from_file_location("spec_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from path: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_lgbm_spec(spec_models_path: Path, yaml_path: Path):
    """Load and validate a LightGBM spec from YAML using the spec models file.

    Returns a Pydantic model instance (LightGBMSpec).
    """
    mod = import_module_from_path(spec_models_path)
    try:
        LightGBMSpec = getattr(mod, "LightGBMSpec")
    except AttributeError as exc:
        raise ImportError("LightGBMSpec not found in spec models module") from exc

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    # Pydantic v2: model_validate; v1 fallback would be parse_obj
    return LightGBMSpec.model_validate(data)


