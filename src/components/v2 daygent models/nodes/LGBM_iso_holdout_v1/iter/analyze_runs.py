#!/usr/bin/env python3
"""
Analyze LightGBM 1D ISO runs: build/refresh history.json (code diffs) and leaderboard.json (metrics).

- Scans artifacts_lgbm_1d_iso/runs/iteration*/ for:
  - results_1d_iso.json
  - code_*.txt (saved code snapshot)

Outputs (in current folder):
  - history.json: per-iteration code changes vs previous iteration, with line-level mapping
  - leaderboard.json: ranked by combined average of test_accuracy and test_auc; includes tie reporting
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import difflib


BASE_DIR = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE_DIR, 'artifacts_lgbm_1d_iso', 'runs')
HISTORY_PATH = os.path.join(BASE_DIR, 'history.json')
LEADERBOARD_PATH = os.path.join(BASE_DIR, 'leaderboard.json')


def list_iteration_dirs(runs_dir: str) -> List[Tuple[int, str, str]]:
    """Return list of (iteration_index, iteration_name, absolute_path), sorted by index."""
    iters: List[Tuple[int, str, str]] = []
    if not os.path.isdir(runs_dir):
        return iters
    for entry in os.listdir(runs_dir):
        path = os.path.join(runs_dir, entry)
        if not os.path.isdir(path):
            continue
        m = re.fullmatch(r"iteration(\d+)", entry)
        if not m:
            continue
        idx = int(m.group(1))
        iters.append((idx, entry, path))
    iters.sort(key=lambda t: t[0])
    return iters


def find_code_snapshot(iter_dir: str) -> Optional[str]:
    """Return absolute path to code_*.txt if present."""
    for fname in os.listdir(iter_dir):
        if fname.startswith('code_') and fname.endswith('.txt'):
            return os.path.join(iter_dir, fname)
    return None


def read_text_lines(path: str) -> Optional[List[str]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().splitlines()
    except Exception:
        return None


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def compute_line_changes(prev_lines: List[str], curr_lines: List[str]) -> Tuple[List[dict], dict]:
    """Return (changes, summary) using difflib opcodes.
    Each change item includes type, previous/new line numbers (1-based) and texts.
    """
    changes: List[dict] = []
    summary = {"replaced": 0, "inserted": 0, "deleted": 0, "total": 0}
    matcher = difflib.SequenceMatcher(None, prev_lines, curr_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'replace':
            # Pairwise map where possible, then leftover as insert/delete
            len_prev = i2 - i1
            len_curr = j2 - j1
            paired = min(len_prev, len_curr)
            for k in range(paired):
                changes.append({
                    "change_type": "replace",
                    "prev_line_no": i1 + k + 1,
                    "prev_text": prev_lines[i1 + k],
                    "new_line_no": j1 + k + 1,
                    "new_text": curr_lines[j1 + k],
                })
                summary["replaced"] += 1
            # Leftover deletions
            for k in range(paired, len_prev):
                changes.append({
                    "change_type": "delete",
                    "prev_line_no": i1 + k + 1,
                    "prev_text": prev_lines[i1 + k],
                    "new_line_no": None,
                    "new_text": "",
                })
                summary["deleted"] += 1
            # Leftover insertions
            for k in range(paired, len_curr):
                changes.append({
                    "change_type": "insert",
                    "prev_line_no": None,
                    "prev_text": "",
                    "new_line_no": j1 + k + 1,
                    "new_text": curr_lines[j1 + k],
                })
                summary["inserted"] += 1
        elif tag == 'delete':
            for k in range(i1, i2):
                changes.append({
                    "change_type": "delete",
                    "prev_line_no": k + 1,
                    "prev_text": prev_lines[k],
                    "new_line_no": None,
                    "new_text": "",
                })
                summary["deleted"] += 1
        elif tag == 'insert':
            for k in range(j1, j2):
                changes.append({
                    "change_type": "insert",
                    "prev_line_no": None,
                    "prev_text": "",
                    "new_line_no": k + 1,
                    "new_text": curr_lines[k],
                })
                summary["inserted"] += 1
    summary["total"] = summary["replaced"] + summary["inserted"] + summary["deleted"]
    return changes, summary


def build_history(iter_dirs: List[Tuple[int, str, str]]) -> dict:
    existing = read_json(HISTORY_PATH) or {}
    entries: List[dict] = []
    prev_code_lines: Optional[List[str]] = None
    prev_iteration_name: Optional[str] = None

    for idx, iter_name, iter_path in iter_dirs:
        code_path = find_code_snapshot(iter_path)
        code_lines = read_text_lines(code_path) if code_path else None
        script_name = os.path.basename(code_path)[len('code_'):-len('.txt')] if code_path else None

        changes: List[dict] = []
        summary = {"replaced": 0, "inserted": 0, "deleted": 0, "total": 0}
        if prev_code_lines is not None and code_lines is not None:
            changes, summary = compute_line_changes(prev_code_lines, code_lines)

        entry = {
            "iteration": iter_name,
            "script_name": script_name,
            "prev_iteration": prev_iteration_name,
            "no_changes": len(changes) == 0,
            "changes": changes,
            "summary": summary,
            "missing_code_snapshot": code_lines is None,
        }
        entries.append(entry)

        prev_code_lines = code_lines if code_lines is not None else prev_code_lines
        prev_iteration_name = iter_name

    history = {
        "generated_at_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "runs_dir": RUNS_DIR,
        "iterations": entries,
    }

    # If existing history has additional metadata, preserve top-level keys not overwritten
    for k, v in existing.items():
        if k not in history:
            history[k] = v

    return history


def safe_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def build_leaderboard(iter_dirs: List[Tuple[int, str, str]]) -> dict:
    rows: List[dict] = []
    for idx, iter_name, iter_path in iter_dirs:
        results_path = os.path.join(iter_path, 'results_1d_iso.json')
        res = read_json(results_path) or {}
        acc = safe_float(res.get('test_accuracy'))
        auc = safe_float(res.get('test_auc'))
        # Combined average of available metrics
        metrics = [m for m in [acc, auc] if m is not None]
        combined = (sum(metrics) / len(metrics)) if metrics else None
        rows.append({
            "iteration": iter_name,
            "test_accuracy": acc,
            "test_auc": auc,
            "combined_average": combined,
            "results_path": results_path,
        })

    # Rank by combined_average desc (dense ranking), None goes last without rank
    ranked = [r for r in rows if r["combined_average"] is not None]
    ranked.sort(key=lambda r: (-r["combined_average"], r["iteration"]))
    rank = 0
    last_score: Optional[float] = None
    ties_map: Dict[float, List[str]] = {}
    for r in ranked:
        score = round(r["combined_average"], 6)
        if last_score is None or score != last_score:
            rank += 1
            last_score = score
        r["rank"] = rank
        ties_map.setdefault(score, []).append(r["iteration"])

    # Attach ranks; mark others as unranked
    for r in rows:
        if "rank" not in r:
            r["rank"] = None

    ties = [iters for score, iters in ties_map.items() if len(iters) > 1]

    leaderboard = {
        "generated_at_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "runs_dir": RUNS_DIR,
        "rows": rows,
        "ties": ties,
        "has_ties": len(ties) > 0,
        "note": "Ranked by average of test_accuracy and test_auc. Missing metrics are ignored in average.",
    }
    return leaderboard


def main() -> None:
    iter_dirs = list_iteration_dirs(RUNS_DIR)

    history = build_history(iter_dirs)
    write_json(HISTORY_PATH, history)

    leaderboard = build_leaderboard(iter_dirs)
    write_json(LEADERBOARD_PATH, leaderboard)

    print(f"Wrote history to: {HISTORY_PATH}")
    print(f"Wrote leaderboard to: {LEADERBOARD_PATH}")


if __name__ == '__main__':
    main()


