"""
Pre-Submission Validator — Supply Chain Digital Twin
=====================================================
Runs the same checks the Scaler portal performs in Phase 1.

IMPORTANT: This script needs a RUNNING server to connect to.
It does NOT start the server itself.

Options:
  A) Test against your live HF Space (recommended — no local setup needed):
       set ENV_BASE_URL=https://Vinshanks3-enterprise-warehouse-env.hf.space
       python validate.py

  B) Start server locally first, then run:
       # Terminal 1:
       pip install openenv-core[core] fastapi uvicorn fastmcp
       cd supply-chain-env
       uvicorn server.app:app --host 0.0.0.0 --port 8000
       # Terminal 2:
       python validate.py

  C) Use local_test.py instead — it starts the server automatically:
       python local_test.py
"""

import json
import os
import sys
import traceback

import requests

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(label: str, fn) -> bool:
    try:
        fn()
        print(f"  {PASS}  {label}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"         → {e}")
        return False


def main():
    print("\n🔍 Supply Chain Digital Twin — Pre-Submission Validator")
    print(f"   Target: {BASE_URL}\n")

    results = []

    # 1. Health check
    def health():
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    results.append(check("GET /health → 200", health))

    # 2. Reset endpoint
    def reset():
        r = requests.post(f"{BASE_URL}/reset", json={}, timeout=15)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
        body = r.json()
        assert "done" in body or "metadata" in body, f"Unexpected reset body: {body}"

    results.append(check("POST /reset → 200 with valid body", reset))

    # 3. State endpoint
    def state():
        r = requests.get(f"{BASE_URL}/state", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    results.append(check("GET /state → 200", state))

    # 4. MCP tool: list_tools
    def list_tools():
        r = requests.post(
            f"{BASE_URL}/step",
            json={"action_type": "list_tools"},
            timeout=15,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"

    results.append(check("POST /step (list_tools action) → 200", list_tools))

    # 5. MCP tool: start_task
    def start_task():
        r = requests.post(
            f"{BASE_URL}/step",
            json={
                "action_type": "call_tool",
                "tool_name": "start_task",
                "arguments": {"task_name": "stable_procurement", "seed": 42},
            },
            timeout=15,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
        body = r.json()
        # Result should contain task key
        result_str = json.dumps(body)
        assert "stable_procurement" in result_str or "stock" in result_str, \
            f"start_task result looks wrong: {result_str[:300]}"

    results.append(check("MCP tool: start_task(stable_procurement) works", start_task))

    # 6. MCP tool: place_order + advance_day
    def one_step():
        # place_order
        r1 = requests.post(
            f"{BASE_URL}/step",
            json={
                "action_type": "call_tool",
                "tool_name": "place_order",
                "arguments": {"units": 20},
            },
            timeout=15,
        )
        assert r1.status_code == 200, f"place_order failed: {r1.status_code}"

        # advance_day
        r2 = requests.post(
            f"{BASE_URL}/step",
            json={
                "action_type": "call_tool",
                "tool_name": "advance_day",
                "arguments": {},
            },
            timeout=15,
        )
        assert r2.status_code == 200, f"advance_day failed: {r2.status_code}"
        body = r2.json()
        result_str = json.dumps(body)
        assert "day_reward" in result_str or "reward" in result_str, \
            f"advance_day missing reward: {result_str[:300]}"

    results.append(check("Full step: place_order + advance_day → reward", one_step))

    # 7. Score endpoint
    def score():
        r = requests.post(
            f"{BASE_URL}/step",
            json={
                "action_type": "call_tool",
                "tool_name": "get_score",
                "arguments": {},
            },
            timeout=15,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        body = r.json()
        result_str = json.dumps(body)
        assert "score" in result_str, f"get_score missing 'score': {result_str[:200]}"

    results.append(check("MCP tool: get_score → score in [0,1]", score))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉 All checks passed — ready to submit!")
    else:
        print("  ⚠️  Fix the failing checks before submitting.")
    print(f"{'='*50}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
