"""Tests for the Server CCode dashboard's pure logic: the access guard and chat matching (no SSH needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ccode_chats import build_chats
from services.ccode_guard import (
    chat_id_ok, client_allowed, effort_ok, model_ok, origin_allowed, path_allowed, window_id_ok,
)

A = "bd2a66cd-d0e6-4393-bfcd-97a83e3855cf"
B = "7d2a4787-1111-2222-3333-444455556666"
C = "aaaa1111-1111-2222-3333-444455556666"


@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1", True), ("::1", True), ("192.168.0.163", False), ("10.0.0.5", False), ("", False), (None, False),
])
def test_only_this_pc_may_connect(host, ok):
    assert client_allowed(host) is ok


@pytest.mark.parametrize("origin,ok", [
    ("http://localhost:5173", True), ("http://127.0.0.1:5173", True), ("http://localhost:5174", True),
    (None, True),                                   # not a browser page (curl on this PC)
    ("https://evil.example.com", False), ("http://localhost.evil.com", False), ("null", False),
    ("http://192.168.0.163:5173", False),
])
def test_only_the_local_page_may_connect(origin, ok):
    assert origin_allowed(origin) is ok


@pytest.mark.parametrize("path,ok", [
    ("/research", True), ("/research/work/x/SESSION.md", True), ("/data/l4", True),
    ("relative/path", False), ("", False), ("/research/\0bad", False),
    ("/home/research/.ssh/authorized_keys", False), ("/home/research/.claude/.credentials.json", False),
    ("/home/research/.claude.json", False), ("/research/../home/research/.ssh/id", False),
])
def test_paths(path, ok):
    assert path_allowed(path) is ok


def test_values_are_validated():
    assert model_ok("opus") and model_ok("fable") and not model_ok("opus; rm -rf /")
    assert effort_ok("xhigh") and effort_ok("max") and not effort_ok("huge")
    assert window_id_ok("@12") and not window_id_ok("@1;ls") and not window_id_ok("0")
    assert chat_id_ok(A) and not chat_id_ok("../x") and not chat_id_ok("abc")


def snapshot():
    return {
        "windows": [
            {"id": "@0", "index": 0, "name": "claude", "pane_pid": 100, "command": "claude"},
            {"id": "@3", "index": 1, "name": "bash", "pane_pid": 200, "command": "claude"},
            {"id": "@4", "index": 2, "name": "bash", "pane_pid": 300, "command": "bash"},
        ],
        "parents": {"100": 50, "201": 200, "200": 50, "300": 50},
        "sessions": [{"pid": 100, "session_id": A, "cwd": "/research", "started": 1},
                     {"pid": 201, "session_id": B, "cwd": "/research", "started": 2}],
        "statuses": {A: {"state": "working", "t": 10}},
        "folders": [f"20261029-2145-{A[:8]}-l4-vs-collector", f"20261029-2140-{B[:8]}"],
        "transcripts": [
            {"session_id": A, "mtime": 30, "size": 9, "title": "first message of A"},
            {"session_id": C, "mtime": 20, "size": 9, "title": "an older chat"},
            {"session_id": "dddd1111-1111-2222-3333-444455556666", "mtime": 40, "size": 9, "title": ""},
        ],
        "guard": "ok",
        "current": "@3",
        "models": {A: "claude-opus-5-5"},
        "context": {A: {"tokens": 382_000, "compactions": 2, "last_compact": 431_330, "t": "x"},
                    B: {"tokens": None, "compactions": 1, "last_compact": 898_339, "t": "y"}},
    }


def test_windows_are_matched_to_chats_and_folders():
    r = build_chats(snapshot())
    a, b, shell = r["running"]
    assert (a["window_id"], a["session_id"], a["name"], a["state"]) == ("@0", A, "l4-vs-collector", "working")
    assert a["folder"] == f"/research/work/20261029-2145-{A[:8]}-l4-vs-collector"
    assert a["model"] == "opus" and b["model"] is None
    assert r["current"] == "@3"
    # claude is a child of the window's shell here; the status file is missing, so it counts as idle
    assert (b["window_id"], b["session_id"], b["state"]) == ("@3", B, "idle")
    assert (shell["kind"], shell["session_id"], shell["name"]) == ("shell", None, "shell")


def test_context_and_compactions_are_passed_on():
    a, b, shell = build_chats(snapshot())["running"]
    assert a["context"] == {"tokens": 382_000, "window": 1_000_000, "pct": 38.2, "compactions": 2, "last_compact": 431_330}
    # just compacted: the new size is unknown until the next reply
    assert b["context"]["tokens"] is None and b["context"]["pct"] is None and b["context"]["compactions"] == 1
    assert shell["context"] is None


def test_older_snapshot_without_context():
    s = snapshot()
    del s["context"]
    assert all(c["context"] is None for c in build_chats(s)["running"])


def test_past_chats_exclude_running_and_empty_ones_newest_first():
    r = build_chats(snapshot())
    assert [p["session_id"] for p in r["past"]] == [C]
    assert r["past"][0]["name"] == "an older chat"
    assert r["past"][0]["folder"] is None


def test_no_tmux_session_yet():
    s = snapshot()
    s["windows"] = None
    r = build_chats(s)
    assert r["running"] == []
    assert {p["session_id"] for p in r["past"]} == {A, C}
    assert r["guard"] == "ok"
