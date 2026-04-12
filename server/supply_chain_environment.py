"""
Supply Chain Digital Twin Environment.

A real-world OpenEnv environment simulating an intelligent warehouse
management system. An AI agent acts as the Chief Supply Chain Officer,
making procurement and inventory decisions across 3 escalating scenarios:

  Task 1 (Easy)   — Stable demand, normal supplier, ~1 week horizon
  Task 2 (Medium) — Volatile demand + seasonal spikes
  Task 3 (Hard)   — Volatile demand + supplier strike + spoilage

All interactions go through MCP tools exposed via FastMCP.
"""

from __future__ import annotations

import random
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = {
    "stable_procurement": {
        "description": (
            "Manage warehouse inventory over 7 days with stable demand (~20 units/day). "
            "Keep stock between 10–150 units, maximize capital. No disruptions."
        ),
        "difficulty": "easy",
        "max_steps": 7,
        "demand_base": 20,
        "demand_variance": 3,
        "spoilage_rate": 0.0,
        "supplier_strike_prob": 0.0,
        "unit_price": 10,
        "sell_price": 18,
        "overstock_penalty": 5,
        "stockout_penalty": 30,
        "initial_stock": 60,
        "initial_capital": 1000,
        "storage_capacity": 150,
    },
    "volatile_demand": {
        "description": (
            "Manage warehouse over 10 days with volatile demand (10–50 units/day) "
            "and periodic demand spikes. Spoilage at 5%/day. Stockouts are costly."
        ),
        "difficulty": "medium",
        "max_steps": 10,
        "demand_base": 28,
        "demand_variance": 18,
        "spoilage_rate": 0.05,
        "supplier_strike_prob": 0.0,
        "unit_price": 10,
        "sell_price": 18,
        "overstock_penalty": 8,
        "stockout_penalty": 40,
        "initial_stock": 80,
        "initial_capital": 1500,
        "storage_capacity": 150,
    },
    "crisis_management": {
        "description": (
            "Manage warehouse over 14 days during a supply-chain crisis. "
            "Demand is highly volatile, spoilage is 8%/day, and a supplier strike "
            "may cut off Product A deliveries for multiple consecutive days. "
            "Survive with capital > 0."
        ),
        "difficulty": "hard",
        "max_steps": 14,
        "demand_base": 30,
        "demand_variance": 22,
        "spoilage_rate": 0.08,
        "supplier_strike_prob": 0.25,
        "unit_price": 10,
        "sell_price": 18,
        "overstock_penalty": 15,
        "stockout_penalty": 50,
        "initial_stock": 100,
        "initial_capital": 2000,
        "storage_capacity": 150,
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyChainEnvironment(MCPEnvironment):
    """
    Intelligent Warehouse Digital Twin.

    Exposes MCP tools:
      - get_state()           → current warehouse snapshot
      - list_tasks()          → enumerate available tasks
      - start_task(task_name) → reset to a specific task
      - place_order(units)    → order stock (may be blocked by strike)
      - advance_day()         → simulate the next day, compute reward
      - get_score()           → 0.0–1.0 score for current episode
    """

    def __init__(self):
        mcp = FastMCP("supply_chain_env")
        self._cfg: dict = {}
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._task_name: str = ""

        # Warehouse state
        self._stock: float = 0.0
        self._capital: float = 0.0
        self._day: int = 0
        self._supplier_strike: bool = False
        self._strike_days_remaining: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._rewards_history: list[float] = []
        self._pending_order: int = 0   # set by place_order, consumed by advance_day
        self._rng: random.Random = random.Random()

        # ------------------------------------------------------------------ #
        # MCP TOOLS
        # ------------------------------------------------------------------ #

        @mcp.tool
        def list_tasks() -> dict:
            """
            List all available tasks with descriptions and difficulty levels.

            Returns:
                Dictionary mapping task_name → {description, difficulty, max_steps}
            """
            return {
                name: {
                    "description": cfg["description"],
                    "difficulty": cfg["difficulty"],
                    "max_steps": cfg["max_steps"],
                }
                for name, cfg in TASKS.items()
            }

        @mcp.tool
        def start_task(task_name: str, seed: int = 42) -> dict:
            """
            Initialise (or restart) the environment for a specific task.

            Args:
                task_name: One of 'stable_procurement', 'volatile_demand', 'crisis_management'
                seed: Random seed for reproducibility

            Returns:
                Initial warehouse state snapshot
            """
            if task_name not in TASKS:
                return {"error": f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"}

            self._cfg = TASKS[task_name]
            self._task_name = task_name
            self._episode_id = str(uuid4())
            self._step_count = 0
            self._stock = float(self._cfg["initial_stock"])
            self._capital = float(self._cfg["initial_capital"])
            self._day = 0
            self._supplier_strike = False
            self._strike_days_remaining = 0
            self._done = False
            self._total_reward = 0.0
            self._rewards_history = []
            self._pending_order = 0
            self._rng = random.Random(seed)

            return self._snapshot("Task started. Place an order then advance_day().")

        @mcp.tool
        def get_state() -> dict:
            """
            Get the current warehouse state.

            Returns:
                Full warehouse snapshot including stock, capital, day, and constraints.
            """
            if not self._task_name:
                return {"error": "No task started. Call start_task(task_name) first."}
            return self._snapshot()

        @mcp.tool
        def place_order(units: int) -> dict:
            """
            Place a purchase order for the next day's delivery.

            Args:
                units: Number of units to order (0–200). Clamped to storage headroom.
                       Blocked if supplier_strike is True.

            Returns:
                Confirmation or error message with updated state.
            """
            if not self._task_name:
                return {"error": "No task started. Call start_task(task_name) first."}
            if self._done:
                return {"error": "Episode is finished. Call start_task() to restart."}

            if self._supplier_strike:
                self._pending_order = 0
                return {
                    "status": "BLOCKED",
                    "reason": "Supplier strike active — no units can be delivered today.",
                    "units_ordered": 0,
                    **self._snapshot(),
                }

            units = max(0, min(units, 200))
            # Clamp to storage headroom
            headroom = int(self._cfg["storage_capacity"] - self._stock)
            units = min(units, max(0, headroom))
            cost = units * self._cfg["unit_price"]

            if cost > self._capital:
                units = int(self._capital // self._cfg["unit_price"])
                cost = units * self._cfg["unit_price"]

            self._pending_order = units
            self._capital -= cost
            self._step_count += 1

            return {
                "status": "OK",
                "units_ordered": units,
                "cost": cost,
                **self._snapshot(f"Order of {units} units placed (cost ${cost:.0f})."),
            }

        @mcp.tool
        def advance_day() -> dict:
            """
            Advance the simulation by one day.

            Sequence:
              1. Deliver pending order (unless strike active)
              2. Roll supplier strike for next day
              3. Apply spoilage
              4. Generate random demand, sell available stock
              5. Compute day reward
              6. Check terminal conditions

            Returns:
                Day summary including demand, spoilage, revenue, reward, and done flag.
            """
            if not self._task_name:
                return {"error": "No task started. Call start_task(task_name) first."}
            if self._done:
                return {"error": "Episode finished. Call start_task() to restart."}

            self._day += 1
            self._step_count += 1
            cfg = self._cfg

            # 1. Deliver order (already deducted from capital in place_order)
            delivered = self._pending_order if not self._supplier_strike else 0
            self._stock = min(self._stock + delivered, cfg["storage_capacity"])
            self._pending_order = 0

            # 2. Update supplier strike status
            if self._strike_days_remaining > 0:
                self._strike_days_remaining -= 1
                self._supplier_strike = self._strike_days_remaining > 0
            else:
                # Roll new strike?
                if self._rng.random() < cfg["supplier_strike_prob"]:
                    self._supplier_strike = True
                    self._strike_days_remaining = self._rng.randint(1, 3)
                else:
                    self._supplier_strike = False

            # 3. Spoilage
            spoiled = int(self._stock * cfg["spoilage_rate"])
            self._stock = max(0.0, self._stock - spoiled)

            # 4. Demand generation (occasional spike for medium/hard)
            demand_base = cfg["demand_base"]
            if cfg["difficulty"] in ("medium", "hard") and self._rng.random() < 0.20:
                demand_base = int(demand_base * 1.8)   # spike
            demand = max(0, int(self._rng.gauss(demand_base, cfg["demand_variance"])))

            units_sold = min(demand, int(self._stock))
            revenue = units_sold * cfg["sell_price"]
            stockout = max(0, demand - units_sold)
            self._stock -= units_sold
            self._capital += revenue

            # 5. Reward computation
            #    Positive: revenue scaled to [0, 1] + capital health bonus
            #    Negative: overstock penalty, stockout penalty
            max_revenue = demand * cfg["sell_price"]
            fill_ratio = units_sold / demand if demand > 0 else 1.0
            revenue_reward = fill_ratio * 0.5                           # 0–0.5

            capital_ratio = min(self._capital / cfg["initial_capital"], 1.0)
            capital_reward = capital_ratio * 0.3                        # 0–0.3

            overstock_penalty = 0.0
            if self._stock > cfg["storage_capacity"] * 0.85:
                overstock_penalty = 0.1 * (self._stock / cfg["storage_capacity"])

            stockout_penalty = 0.0
            if stockout > 0:
                stockout_penalty = 0.15 * min(stockout / demand, 1.0) if demand > 0 else 0

            day_reward = revenue_reward + capital_reward - overstock_penalty - stockout_penalty
            day_reward = max(-0.5, min(1.0, day_reward))

            self._total_reward += day_reward
            self._rewards_history.append(round(day_reward, 4))

            # 6. Terminal check
            done = (
                self._day >= cfg["max_steps"]
                or self._capital <= 0
            )
            self._done = done

            return {
                "day": self._day,
                "delivered": delivered,
                "spoiled": spoiled,
                "demand": demand,
                "units_sold": units_sold,
                "stockout": stockout,
                "revenue": round(revenue, 2),
                "day_reward": round(day_reward, 4),
                "done": done,
                "supplier_strike_active": self._supplier_strike,
                **self._snapshot(),
            }

        @mcp.tool
        def get_score() -> dict:
            """
            Compute the final normalised score for the current episode (0.0–1.0).

            Score components:
              - Capital survival  (40%): capital_final / capital_initial, clamped [0,1]
              - Fill rate         (30%): avg day_reward contribution
              - Completion bonus  (30%): 1.0 if all steps completed without bankruptcy

            Returns:
                {score, capital_ratio, avg_reward, episode_complete, task_name}
            """
            if not self._task_name:
                return {"error": "No task started."}

            cfg = self._cfg
            capital_ratio = min(max(self._capital / cfg["initial_capital"], 0.0), 1.0)
            avg_reward = (
                sum(self._rewards_history) / len(self._rewards_history)
                if self._rewards_history else 0.0
            )
            # Normalise avg_reward from [-0.5, 1.0] → [0, 1]
            norm_avg = (avg_reward + 0.5) / 1.5

            completed = self._day >= cfg["max_steps"] and self._capital > 0
            completion_bonus = 1.0 if completed else max(0.0, self._day / cfg["max_steps"] - 0.1)

            score = (
                0.40 * capital_ratio
                + 0.30 * max(0.0, norm_avg)
                + 0.30 * completion_bonus
            )
            score = round(min(max(score, 0.0), 1.0), 4)

            return {
                "score": score,
                "capital_ratio": round(capital_ratio, 4),
                "avg_day_reward": round(avg_reward, 4),
                "episode_complete": completed,
                "task_name": self._task_name,
                "days_survived": self._day,
                "max_days": cfg["max_steps"],
                "final_capital": round(self._capital, 2),
                "final_stock": round(self._stock, 2),
            }

        super().__init__(mcp)
        self._state_obj = State(episode_id=self._episode_id, step_count=0)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _snapshot(self, message: str = "") -> dict:
        cfg = self._cfg
        snap = {
            "task": self._task_name,
            "day": self._day,
            "max_days": cfg.get("max_steps", 0),
            "stock": round(self._stock, 1),
            "capital": round(self._capital, 2),
            "storage_capacity": cfg.get("storage_capacity", 150),
            "supplier_strike": self._supplier_strike,
            "done": self._done,
            "step_count": self._step_count,
        }
        if message:
            snap["message"] = message
        return snap

    # ---------------------------------------------------------------------- #
    # OpenEnv interface
    # ---------------------------------------------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._state_obj = State(episode_id=self._episode_id, step_count=0)
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": (
                    "Supply Chain Digital Twin ready. "
                    "Call list_tasks() to see available tasks, "
                    "then start_task(task_name) to begin."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=self._done,
            reward=self._total_reward,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools."},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1
        self._state_obj.step_count = self._step_count
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1
        self._state_obj.step_count = self._step_count
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        self._state_obj.step_count = self._step_count
        return self._state_obj
