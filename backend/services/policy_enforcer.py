from __future__ import annotations

"""
Policy enforcer scaffold

Two main checks:
 1) check_static(spec, run_cfg) → list[str]
    Pre-run validation against the node's YAML spec (validation style, thresholding
    style, and iteration budgets).

 2) check_postrun(spec, results_json_path) → list[str]
    Post-run validation to ensure the trainer produced expected metrics and that the
    chosen threshold lies within declared bounds.

Return value contract:
  - [] (empty list) means COMPLIANT
  - non-empty list of strings means violations (NON-COMPLIANT)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import json


def _threshold_bounds(spec) -> Optional[Tuple[float, float]]:
    """Return (low, high) for thresholding according to the spec, if present.

    Exactly one of these should be defined by the spec:
      - eval_protocol.threshold_grid
      - eval_protocol.threshold_selection
    """
    tg = getattr(spec.eval_protocol, "threshold_grid", None)
    ts = getattr(spec.eval_protocol, "threshold_selection", None)
    if tg is not None and ts is None:
        return float(tg.low), float(tg.high)
    if ts is not None and tg is None:
        return float(ts.low), float(ts.high)
    return None


def check_static(spec, run_cfg: Dict[str, Any]) -> List[str]:
    """Pre-run static policy checks.

    - Enforce validation method (holdout vs purged_cv)
    - Enforce thresholding style (grid vs selection)
    - Enforce iteration budgets (time, trials, code-change size)
    """
    violations: List[str] = []

    # Validation method must be one of the supported options
    v = getattr(spec.eval_protocol, "validation", None)
    method = getattr(v, "method", None) if v else None
    if method not in {"holdout_time_ordered", "purged_time_series_cv"}:
        violations.append(f"Unsupported validation method: {method}")

    # Exactly one thresholding style should be configured
    tg = getattr(spec.eval_protocol, "threshold_grid", None)
    ts = getattr(spec.eval_protocol, "threshold_selection", None)
    if (tg is None and ts is None) or (tg is not None and ts is not None):
        violations.append("Exactly one thresholding style must be set (grid OR selection).")

    # Node-specific validation parameter checks
    v_obj = getattr(spec.eval_protocol, "validation", None)
    if method == "holdout_time_ordered" and v_obj is not None:
        # Require sane ratios and they should sum to ~1
        train_ratio = float(getattr(v_obj, "train_ratio", 0.0))
        val_ratio = float(getattr(v_obj, "val_ratio", 0.0))
        if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
            violations.append(f"Holdout ratios must be in (0,1): train={train_ratio}, val={val_ratio}")
        if not math.isclose(train_ratio + val_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            violations.append(f"Holdout ratios must sum to 1.0 (got {train_ratio + val_ratio:.6f})")

    if method == "purged_time_series_cv" and v_obj is not None:
        # Guard CV parameters (n_splits reasonable, gap_days within allowed range; recency weighting enabled)
        n_splits = int(getattr(v_obj, "n_splits", 0))
        gap_days = int(getattr(v_obj, "gap_days", -1))
        rec_w = getattr(v_obj, "recency_weighting", None)
        if n_splits < 2 or n_splits > 20:
            violations.append(f"CV n_splits out of bounds: {n_splits} (expected 2..20)")
        # From your notes, acceptable gap days is roughly [3,10]
        if gap_days < 3 or gap_days > 10:
            violations.append(f"CV gap_days {gap_days} outside [3,10]")
        if not isinstance(rec_w, dict) or not rec_w:
            violations.append("CV recency_weighting must be provided and non-empty")
        else:
            enabled_val = rec_w.get("enabled", False)
            # treat any truthy value as enabled
            if not bool(enabled_val):
                violations.append("CV recency_weighting.enabled must be true")
            tau_bars = rec_w.get("tau_bars", None)
            try:
                tau_bars_f = float(tau_bars)
                # from registry notes: [120,240]
                if tau_bars_f < 100 or tau_bars_f > 300:
                    violations.append(f"CV recency_weighting.tau_bars {tau_bars_f} looks out of an expected range (~120..240)")
            except Exception:
                violations.append("CV recency_weighting.tau_bars must be numeric")

    # Budgets
    change_budget = int(getattr(spec, "change_budget_per_loop", 0))
    trials_budget = int(getattr(spec, "trials_budget_per_loop", 0))
    time_budget = int(getattr(spec, "time_budget_minutes", 0))

    if "diff_line_count" in run_cfg:
        try:
            if int(run_cfg["diff_line_count"]) > change_budget:
                violations.append(
                    f"Change budget exceeded: {run_cfg['diff_line_count']} > {change_budget}"
                )
        except Exception:
            violations.append("Invalid run_cfg.diff_line_count (must be int)")

    if "trials" in run_cfg:
        try:
            if int(run_cfg["trials"]) > trials_budget:
                violations.append(
                    f"Trials budget exceeded: {run_cfg['trials']} > {trials_budget}"
                )
        except Exception:
            violations.append("Invalid run_cfg.trials (must be int)")

    if "time_limit_minutes" in run_cfg:
        try:
            if int(run_cfg["time_limit_minutes"]) > time_budget:
                violations.append(
                    f"Time budget exceeded: {run_cfg['time_limit_minutes']} > {time_budget}"
                )
        except Exception:
            violations.append("Invalid run_cfg.time_limit_minutes (must be int)")

    return violations


def check_postrun(spec, results_json_path: Path) -> List[str]:
    """Post-run policy checks against trainer outputs (results JSON).

    - Holdout nodes: require validation metrics and ensure chosen_threshold within bounds
    - CV nodes: require cv metrics and ensure chosen_threshold within bounds
    - Both: require test metrics (accuracy and AUC)
    """
    violations: List[str] = []

    try:
        results = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    except Exception as e:
        return [f"Results file not readable: {e}"]

    # Common: threshold bound check (if present)
    bounds = _threshold_bounds(spec)
    if bounds is not None and "chosen_threshold" in results and results["chosen_threshold"] is not None:
        low, high = bounds
        try:
            thr = float(results["chosen_threshold"])
            if thr < low or thr > high:
                violations.append(
                    f"chosen_threshold {thr:.3f} outside [{low:.3f}, {high:.3f}]"
                )
            # Optional: grid step alignment when threshold_grid is used
            tg = getattr(spec.eval_protocol, "threshold_grid", None)
            if tg is not None:
                step = float(getattr(tg, "step", 0.01))
                # Check that (thr - low) is approximately a multiple of step
                k = round((thr - low) / step)
                aligned = math.isclose(thr, low + k * step, rel_tol=1e-9, abs_tol=1e-9)
                if not aligned:
                    violations.append(
                        f"chosen_threshold {thr:.2f} is not aligned to grid step {step} starting at {low}"
                    )
        except Exception:
            violations.append("chosen_threshold is not a number")

    # Method-specific metric requirements
    v = getattr(spec.eval_protocol, "validation", None)
    method = getattr(v, "method", None) if v else None

    if method == "holdout_time_ordered":
        if (
            "validation_accuracy" not in results
            and "validation_auc" not in results
        ):
            violations.append(
                "Holdout metrics missing (validation_accuracy or validation_auc)."
            )
    elif method == "purged_time_series_cv":
        for key in ("cv_mean_auc", "cv_mean_f1", "cv_best_iter_median"):
            if key not in results:
                violations.append(f"CV metric missing: {key}")
        # cv_best_iter_median should be a positive integer
        try:
            if int(results.get("cv_best_iter_median", 0)) <= 0:
                violations.append("cv_best_iter_median must be > 0")
        except Exception:
            violations.append("cv_best_iter_median must be an integer")
    else:
        violations.append(f"Unknown validation method: {method}")

    # Common test metrics (both trainers write these fields)
    for key in ("test_accuracy", "test_auc"):
        if key not in results:
            violations.append(f"Test metric missing: {key}")

    # Feature names sanity check: expect the 16-D vector length for these ISO nodes
    feats = results.get("feature_names")
    if isinstance(feats, list) and len(feats) != 16:
        violations.append(f"feature_names length {len(feats)} != 16 (expected 16 for ISO 1D)")

    return violations


