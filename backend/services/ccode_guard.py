"""Access rules for the Server CCode dashboard.

The dashboard drives Claude Code on the Tokyo server with permissions bypassed, and this backend listens on every
network address for the other dashboards. So every /api/ccode route accepts only this PC, and only the local Daygent
page (browsers do not apply CORS to WebSockets, so the Origin has to be checked here). All values that reach a remote
command are validated against fixed lists or strict patterns.
"""
import posixpath
import re
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "::1"}
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1"}
MODELS = {"fable", "opus", "sonnet", "haiku"}
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# sign-in secrets of the research user: never shown or downloaded, even though the account can read them
SECRET_PATHS = ("/home/research/.ssh", "/home/research/.claude/.credentials.json", "/home/research/.claude.json")


def client_allowed(host):
    return host in LOOPBACK


def origin_allowed(origin):
    """A browser page always sends its Origin; no Origin means a local tool such as curl (already loopback-checked)."""
    if origin is None:
        return True
    try:
        u = urlparse(origin)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and u.hostname in LOCAL_HOSTNAMES


def path_allowed(path):
    if not path or not path.startswith("/") or "\0" in path:
        return False
    p = posixpath.normpath(path)
    return not any(p == s or p.startswith(s + "/") for s in SECRET_PATHS)


def model_ok(model):
    return model in MODELS


def effort_ok(effort):
    return effort in EFFORTS


def window_id_ok(window_id):
    return bool(re.fullmatch(r"@\d+", window_id or ""))


def chat_id_ok(chat_id):
    return bool(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", chat_id or ""))
