from __future__ import annotations

from pathlib import Path
import yaml
from fastapi import APIRouter, HTTPException

from services.spec_loader import load_lgbm_spec

router = APIRouter(prefix="/api/training", tags=["nodes"])


def _root() -> Path:
    # backend/routers/nodes.py -> backend/ -> project root
    return Path(__file__).resolve().parents[1].parent


def _spec_models_path() -> Path:
    root = _root()
    return (
        root
        / "src"
        / "components"
        / "v2 daygent models"
        / "nodes"
        / "pydantic families"
        / "lgbm_spy1d"
        / "spec_models.py"
    )


def _node_yaml_path(node_id: str) -> Path:
    root = _root()
    # Expect node folder name equals node_id
    yaml_name = f"{node_id}.yaml"
    path = (
        root
        / "src"
        / "components"
        / "v2 daygent models"
        / "nodes"
        / node_id
        / yaml_name
    )
    if not path.exists():
        # allow node YAML file not matching folder (fallback common filenames)
        candidates = list((root / "src" / "components" / "v2 daygent models" / "nodes" / node_id).glob("*.yaml"))
        if not candidates:
            raise HTTPException(status_code=404, detail=f"YAML for node {node_id} not found")
        return candidates[0]
    return path


@router.get("/nodes/{node_id}/spec")
def get_node_spec(node_id: str):
    spec_models_path = _spec_models_path()
    yaml_path = _node_yaml_path(node_id)
    try:
        spec = load_lgbm_spec(spec_models_path, yaml_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Spec load/validate failed: {e}")
    # Return as plain dict for JSON
    return spec.model_dump()


def _nodes_dir() -> Path:
    root = _root()
    return (
        root
        / "src"
        / "components"
        / "v2 daygent models"
        / "nodes"
    )


def _find_yaml_in_dir(node_dir: Path) -> Path | None:
    cands = list(node_dir.glob("*.yaml"))
    return cands[0] if cands else None


@router.get("/nodes")
def list_nodes():
    nodes_dir = _nodes_dir()
    spec_models_path = _spec_models_path()
    if not nodes_dir.exists():
        return []

    summaries = []
    for child in nodes_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower().strip() == "pydantic families":
            continue
        yaml_path = _find_yaml_in_dir(child)
        if yaml_path is None:
            continue

        # Load YAML minimally to detect family
        raw = {}
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass

        family = str(raw.get("family", "")).strip()

        # Default summary values
        summary = {
            "node_id": raw.get("node_id", child.name),
            "family": family or "unknown",
            "description": raw.get("description", ""),
            "validation_method": None,
            "thresholding": None,
            "change_budget_per_loop": raw.get("change_budget_per_loop"),
            "trials_budget_per_loop": raw.get("trials_budget_per_loop"),
            "time_budget_minutes": raw.get("time_budget_minutes"),
            "yaml_path": str(yaml_path),
        }

        # If LightGBM, load typed spec for robust summary
        if family == "LightGBMClassifier":
            try:
                spec = load_lgbm_spec(spec_models_path, yaml_path)
                v = getattr(spec.eval_protocol, "validation", None)
                validation_method = getattr(v, "method", None) if v else None
                tg = getattr(spec.eval_protocol, "threshold_grid", None)
                ts = getattr(spec.eval_protocol, "threshold_selection", None)
                thresholding = (
                    "accuracy_grid" if tg is not None else (
                        "median_F1_across_folds" if ts is not None else None
                    )
                )

                summary.update({
                    "node_id": spec.node_id,
                    "description": spec.description,
                    "validation_method": validation_method,
                    "thresholding": thresholding,
                    "change_budget_per_loop": spec.change_budget_per_loop,
                    "trials_budget_per_loop": spec.trials_budget_per_loop,
                    "time_budget_minutes": spec.time_budget_minutes,
                })
            except Exception:
                # Fall back to raw YAML-derived summary
                pass

        summaries.append(summary)

    # Sort by node_id for stable output
    summaries.sort(key=lambda s: str(s.get("node_id", "")))
    return summaries


