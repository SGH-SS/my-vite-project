from __future__ import annotations

from pathlib import Path
from backend.services.spec_loader import load_lgbm_spec


ROOT = Path(__file__).resolve().parents[2]

SPEC_MODELS_PATH = (
    ROOT
    / "src"
    / "components"
    / "v2 daygent models"
    / "nodes"
    / "pydantic families"
    / "lgbm_spy1d"
    / "spec_models.py"
)

HOLDOUT_YAML = (
    ROOT
    / "src"
    / "components"
    / "v2 daygent models"
    / "nodes"
    / "LGBM_iso_holdout_v1"
    / "LGBM_iso_holdout_v1.yaml"
)

CV_YAML = (
    ROOT
    / "src"
    / "components"
    / "v2 daygent models"
    / "nodes"
    / "LGBM_iso_cv_weighted_v1"
    / "LGBM_iso_cv_weighted_v1.yaml"
)


def assert_holdout(spec):
    assert spec.family == "LightGBMClassifier"
    assert spec.eval_protocol.validation and spec.eval_protocol.validation.method == "holdout_time_ordered"
    assert spec.eval_protocol.threshold_grid is not None
    assert spec.eval_protocol.threshold_selection is None
    tg = spec.eval_protocol.threshold_grid
    assert 0.0 <= tg.low <= 1.0
    assert 0.0 <= tg.high <= 1.0
    assert 0.0 < tg.step <= 1.0


def assert_cv(spec):
    assert spec.family == "LightGBMClassifier"
    v = spec.eval_protocol.validation
    assert v and v.method == "purged_time_series_cv"
    assert v.n_splits >= 2 and v.gap_days >= 0
    assert spec.eval_protocol.threshold_selection is not None
    assert spec.eval_protocol.threshold_grid is None
    ts = spec.eval_protocol.threshold_selection
    assert ts.metric == "F1" and ts.aggregation == "median_across_folds"
    assert 0.0 <= ts.low <= 1.0
    assert 0.0 <= ts.high <= 1.0
    assert 0.0 < ts.step <= 1.0


def main():
    holdout = load_lgbm_spec(SPEC_MODELS_PATH, HOLDOUT_YAML)
    cv = load_lgbm_spec(SPEC_MODELS_PATH, CV_YAML)

    assert_holdout(holdout)
    assert_cv(cv)

    print("[OK] Holdout:", holdout.node_id, "validation:", holdout.eval_protocol.validation.method)
    print("[OK] CV:", cv.node_id, "validation:", cv.eval_protocol.validation.method)
    print("[OK] Budgets:", {
        "holdout": (holdout.change_budget_per_loop, holdout.trials_budget_per_loop, holdout.time_budget_minutes),
        "cv": (cv.change_budget_per_loop, cv.trials_budget_per_loop, cv.time_budget_minutes),
    })
    print("[OK] Similarity guard thresholds:", holdout.similarity_guard.threshold, cv.similarity_guard.threshold)


if __name__ == "__main__":
    main()


