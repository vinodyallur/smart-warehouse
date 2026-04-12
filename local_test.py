"""
local_test.py — Starts the server in-process and validates all Phase 1 checks.

Run from inside the supply-chain-env folder:
    pip install openenv-core[core] fastapi uvicorn requests fastmcp
    python local_test.py
"""

import json
import sys
import threading
import time

import requests
import uvicorn

BASE_URL = "http://localhost:8765"   # use a non-standard port to avoid conflicts

PASS = "✅ PASS"
FAIL = "❌ FAIL"


# ── Start server in a background thread ─────────────────────────────────────
def start_server():
    try:
        from server.app import app
    except ImportError:
        # If running from repo root
        import importlib, pathlib, sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from server.app import app

    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="error")
    server = uvicorn.Server(config)
    server.run()


def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Checks ───────────────────────────────────────────────────────────────────
def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"         → {e}")
        return False


def run_checks():
    results = []

    def health():
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    results.append(check("GET /health → 200", health))

    def reset():
        r = requests.post(f"{BASE_URL}/reset", json={}, timeout=15)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        body = r.json()
        assert "done" in body or "metadata" in body, f"Bad reset body: {body}"
    results.append(check("POST /reset → 200 with valid body", reset))

    def state():
        r = requests.get(f"{BASE_URL}/state", timeout=10)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    results.append(check("GET /state → 200", state))

    def list_tools():
        r = requests.post(f"{BASE_URL}/step",
                          json={"action_type": "list_tools"}, timeout=15)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    results.append(check("POST /step (list_tools) → 200", list_tools))

    def start_task():
        r = requests.post(f"{BASE_URL}/step", json={
            "action_type": "call_tool",
            "tool_name": "start_task",
            "arguments": {"task_name": "stable_procurement", "seed": 42},
        }, timeout=15)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        txt = json.dumps(r.json())
        assert "stock" in txt or "stable" in txt, f"Unexpected: {txt[:200]}"
    results.append(check("start_task(stable_procurement) via MCP", start_task))

    def one_step():
        requests.post(f"{BASE_URL}/step", json={
            "action_type": "call_tool", "tool_name": "start_task",
            "arguments": {"task_name": "stable_procurement", "seed": 1},
        }, timeout=15)
        r1 = requests.post(f"{BASE_URL}/step", json={
            "action_type": "call_tool", "tool_name": "place_order",
            "arguments": {"units": 20},
        }, timeout=15)
        assert r1.status_code == 200, f"place_order HTTP {r1.status_code}"
        r2 = requests.post(f"{BASE_URL}/step", json={
            "action_type": "call_tool", "tool_name": "advance_day",
            "arguments": {},
        }, timeout=15)
        assert r2.status_code == 200, f"advance_day HTTP {r2.status_code}"
        txt = json.dumps(r2.json())
        assert "reward" in txt or "day" in txt, f"Missing reward: {txt[:200]}"
    results.append(check("place_order + advance_day → reward in response", one_step))

    def score():
        r = requests.post(f"{BASE_URL}/step", json={
            "action_type": "call_tool", "tool_name": "get_score",
            "arguments": {},
        }, timeout=15)
        assert r.status_code == 200, f"HTTP {r.status_code}"
        txt = json.dumps(r.json())
        assert "score" in txt, f"Missing score: {txt[:200]}"
    results.append(check("get_score() → score present", score))

    # 3 tasks reachable
    for task in ["stable_procurement", "volatile_demand", "crisis_management"]:
        def task_check(t=task):
            r = requests.post(f"{BASE_URL}/step", json={
                "action_type": "call_tool", "tool_name": "start_task",
                "arguments": {"task_name": t, "seed": 0},
            }, timeout=15)
            assert r.status_code == 200, f"HTTP {r.status_code}"
            txt = json.dumps(r.json())
            assert "error" not in txt.lower() or "stock" in txt, f"Error in response: {txt[:200]}"
        results.append(check(f"Task '{task}' accessible", task_check))

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*52}")
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉 All checks passed — safe to submit!")
    else:
        print("  ⚠️  Fix failing checks before submitting.")
    print(f"{'='*52}\n")
    return passed == total


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Starting server on port 8765…")
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    if not wait_for_server(timeout=30):
        print("❌ Server did not start within 30 seconds.")
        print("   Make sure dependencies are installed:")
        print("   pip install openenv-core[core] fastapi uvicorn requests fastmcp")
        sys.exit(1)

    print("✅ Server is up. Running checks…\n")
    ok = run_checks()
    sys.exit(0 if ok else 1)
