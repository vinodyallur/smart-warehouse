---
title: Supply Chain Digital Twin
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🏭 Supply Chain Digital Twin — Intelligent Warehouse Management Environment

> **OpenEnv Round 1 Submission** | Team: Vinod Yallur & Srushti R M

## Problem Statement

Global supply chains lose **$1.5 trillion annually** from poor inventory decisions, stockouts, and unplanned disruptions. Warehouse managers face an impossible balancing act: maintain enough stock to meet unpredictable demand while avoiding expensive overstock — all while navigating real-world shocks like supplier strikes, spoilage, and demand spikes.

This environment simulates that exact challenge. An AI agent acts as the **Chief Supply Chain Officer**, making daily procurement decisions in a realistic warehouse under increasingly severe market conditions.

---

## What This Environment Simulates

The **Supply Chain Digital Twin** models a perishable-goods warehouse with:

- **Non-linear stochastic demand** — Gaussian base demand with random spike events
- **Perishable inventory** — Stock spoils at 5–8% per day (task-dependent)
- **Supplier disruptions** — Random multi-day strikes block all deliveries
- **Capital constraints** — Every order costs money; stockouts and overstock carry penalties
- **Storage limits** — 150-unit warehouse capacity enforces lean inventory discipline

---

## Tasks

| Task | Difficulty | Days | Key Challenge |
|------|-----------|------|---------------|
| `stable_procurement` | 🟢 Easy | 7 | Basic inventory management, no disruptions |
| `volatile_demand` | 🟡 Medium | 10 | Demand spikes + 5% daily spoilage |
| `crisis_management` | 🔴 Hard | 14 | Supplier strikes + 8% spoilage + high volatility |

All tasks require the agent to keep capital > 0 and stock between 0–150 units.

---

## Action Space

The agent interacts via **MCP tools**:

| Tool | Arguments | Description |
|------|-----------|-------------|
| `list_tasks()` | — | Enumerate available tasks |
| `start_task(task_name, seed)` | `task_name: str`, `seed: int` | Initialise episode |
| `get_state()` | — | Full warehouse snapshot |
| `place_order(units)` | `units: int` | Purchase stock (blocked during strikes) |
| `advance_day()` | — | Simulate one day: deliver → spoil → demand → reward |
| `get_score()` | — | Normalised episode score [0.0–1.0] |

---

## Observation Space

Each `advance_day()` returns:

```json
{
  "day": 5,
  "delivered": 30,
  "spoiled": 4,
  "demand": 28,
  "units_sold": 28,
  "stockout": 0,
  "revenue": 504.0,
  "day_reward": 0.7123,
  "done": false,
  "supplier_strike_active": false,
  "stock": 98.0,
  "capital": 1840.0
}
```

---

## Reward Function

The reward is **dense** (computed every day, not just at episode end):

```
day_reward = (
    fill_ratio       × 0.50   # Revenue efficiency: units_sold / demand
  + capital_ratio    × 0.30   # Capital health: capital / initial_capital
  - overstock_penalty         # Up to 0.10 when stock > 85% capacity
  - stockout_penalty          # Up to 0.15 proportional to unmet demand
)
```

Clamped to `[-0.50, 1.00]`.

**Final episode score** (0.0–1.0):

```
score = 0.40 × capital_ratio + 0.30 × norm_avg_reward + 0.30 × completion_bonus
```

---

## Baseline Scores (Qwen-2.5-72B)

| Task | Score | Notes |
|------|-------|-------|
| `stable_procurement` | ~0.72 | Consistent ordering, low stockout rate |
| `volatile_demand` | ~0.58 | Occasional over-ordering during spikes |
| `crisis_management` | ~0.51 | Strike adaptation, some capital drain |
| **Average** | **~0.60** | |

---

## Quick Start

### Using the HuggingFace Space

```python
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient(base_url="https://your-space.hf.space") as env:
    env.reset()
    env.call_tool("start_task", task_name="stable_procurement", seed=42)
    env.call_tool("place_order", units=30)
    result = env.call_tool("advance_day")
    print(result)
```

### Run Inference Script

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

### Run Command Center UI

```bash
streamlit run app_ui.py
```

### Docker

```bash
docker build -t supply-chain-env:latest .
docker run -p 8000:8000 supply-chain-env:latest
```

---

## Project Structure

```
supply-chain-env/
├── __init__.py                    # Package exports
├── client.py                      # SupplyChainEnv MCP client
├── inference.py                   # Baseline inference script (mandatory format)
├── app_ui.py                      # Streamlit Command Center dashboard
├── openenv.yaml                   # OpenEnv spec metadata
├── pyproject.toml                 # Dependencies
├── Dockerfile                     # Container image
├── README.md                      # This file
└── server/
    ├── __init__.py
    ├── app.py                     # FastAPI application
    └── supply_chain_environment.py  # Core RL environment
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | HF Inference | LLM API endpoint |
| `MODEL_NAME` | Yes | Qwen-2.5-72B | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace API token |
| `ENV_BASE_URL` | No | `localhost:8000` | Environment server URL |

---

## Why This Matters

Supply chain optimisation is a **$900B+ market**. Current approaches use hand-crafted heuristics (reorder points, safety stock formulas) that fail under disruptions. An LLM agent that can reason about demand uncertainty, strike risk, and spoilage trade-offs represents a genuine step toward **autonomous supply chain management** — a real-world problem with measurable, high-stakes outcomes.
