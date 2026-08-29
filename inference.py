"""
Inference Script — Supply Chain Digital Twin
============================================

Mandatory environment variables:
    API_BASE_URL    LLM endpoint (OpenAI-compatible)
    MODEL_NAME      Model identifier
    HF_TOKEN        Hugging Face / API token
    ENV_BASE_URL    HF Space URL (default: http://localhost:8000)

Stdout format (mandatory):
    [START] task=<task_name> env=supply_chain_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys

from openai import OpenAI
from openenv.core.mcp_client import MCPToolClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS = ["stable_procurement", "volatile_demand", "crisis_management"]

SYSTEM_PROMPT = """You are an expert Chief Supply Chain Officer managing a warehouse.
Your goal is to maximise revenue while keeping inventory healthy and capital positive.

Rules:
- Storage capacity: 150 units maximum
- Ordering costs money upfront; selling generates revenue
- Spoilage destroys stock each day
- Supplier strikes block all deliveries
- Stockouts (demand > stock) are expensive penalties
- Overstock near capacity triggers penalties

Each turn you MUST call exactly one tool: either place_order(units=N) or advance_day().
The episode follows this rhythm: place_order → advance_day → place_order → advance_day …

Strategy tips:
- On strike days: order 0 (call place_order with units=0) before advance_day
- When stock is low: order aggressively if capital allows
- When stock is near 150: order very little to avoid overstock penalty
- Monitor capital: never let it drop to 0

Before each order, provide a 1-sentence Strategic Reasoning in your response
explaining your choice based on current stock, spoilage risks, or potential strikes."""


def make_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy",
    )


def run_task(task_name: str, client: OpenAI, env: MCPToolClient, seed: int = 42) -> dict:
    """Run one full episode for a task. Returns {score, steps, rewards, success}."""
    rewards: list[float] = []
    step = 0
    last_error = None
    done = False
    success = False

    print(f"[START] task={task_name} env=supply_chain_env model={MODEL_NAME}", flush=True)

    # ---- Reset environment and start task ----
    env.reset()
    start_result = env.call_tool("start_task", task_name=task_name, seed=seed)

    # Fetch available tools for the LLM
    tools_list = env.list_tools()
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema if hasattr(t, "inputSchema") else {},
            },
        }
        for t in tools_list
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task '{task_name}' started. Current state:\n"
                f"{json.dumps(start_result, indent=2)}\n\n"
                "Begin managing the warehouse. Call place_order first, then advance_day."
            ),
        },
    ]

    max_steps = 80  # safety cap (14 days × ~3 calls each + buffer)

    while step < max_steps and not done:
        try:
            response = client.chat.completions.createPayment(
                model=MODEL_NAME,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=512,
                temperature=0.3,
            )

            choice = response.choices[0]
            msg = choice.message

            # Append assistant message
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

            if not msg.tool_calls:
                # LLM gave text only — nudge it
                messages.append({
                    "role": "user",
                    "content": "Please call a tool: place_order(units=N) or advance_day().",
                })
                continue

            # Execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                step += 1
                action_str = f"{tool_name}({', '.join(f'{k}={v}' for k,v in args.items())})"

                try:
                    result = env.call_tool(tool_name, **args)
                    last_error = result.get("error") if isinstance(result, dict) else None
                except Exception as exc:
                    result = {"error": str(exc)}
                    last_error = str(exc)

                # Extract reward/done from advance_day results
                step_reward = 0.0
                if isinstance(result, dict):
                    if "day_reward" in result:
                        step_reward = result["day_reward"]
                        rewards.append(step_reward)
                    done = result.get("done", False)

                err_str = last_error if last_error else "null"
                done_str = "true" if done else "false"
                print(
                    f"[STEP]  step={step} action={action_str} "
                    f"reward={step_reward:.2f} done={done_str} error={err_str}",
                    flush=True,
                )

                # Feed result back to LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

                if done:
                    break

        except Exception as exc:
            last_error = str(exc)
            print(
                f"[STEP]  step={step} action=error reward=0.00 done=false error={exc}",
                flush=True,
            )
            break

    # ---- Fetch final score ----
    score = 0.0
    try:
        score_result = env.call_tool("get_score")
        if isinstance(score_result, dict):
            score = float(score_result.get("score", 0.0))
            success = score_result.get("episode_complete", False)
    except Exception:
        pass

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str = "true" if success else "false"
    print(
        f"[END]   success={success_str} steps={step} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )

    return {"score": score, "steps": step, "rewards": rewards, "success": success}


def main():
    client = make_client()

    all_scores = []
    try:
        with MCPToolClient(base_url=ENV_BASE_URL) as env:
            for task_name in TASKS:
                result = run_task(task_name, client, env)
                all_scores.append(result["score"])
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\nOverall average score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
