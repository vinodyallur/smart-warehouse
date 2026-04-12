"""
Supply Chain Digital Twin — Action & Observation Models
=======================================================

Typed Pydantic models for the OpenEnv spec.
The environment uses MCP tools, so Actions are tool calls and
Observations carry the tool result plus reward/done signals.

These are re-exported from the top-level package for convenience:
    from supply_chain_env import WarehouseAction, WarehouseObservation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action models
# ---------------------------------------------------------------------------

class WarehouseAction(BaseModel):
    """
    An action sent to the Supply Chain Digital Twin.

    The environment is MCP-based: every action is a tool call.
    Set tool_name to one of:
      - "list_tasks"
      - "start_task"
      - "get_state"
      - "place_order"
      - "advance_day"
      - "get_score"

    Arguments are passed as key-value pairs in `arguments`.
    """

    tool_name: str = Field(
        ...,
        description=(
            "MCP tool to call. One of: list_tasks, start_task, "
            "get_state, place_order, advance_day, get_score."
        ),
        examples=["place_order", "advance_day"],
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the tool.",
        examples=[{"units": 30}, {"task_name": "crisis_management", "seed": 42}],
    )

    model_config = {"extra": "allow"}


class PlaceOrderAction(BaseModel):
    """Convenience model for the place_order tool."""
    tool_name: str = Field(default="place_order", frozen=True)
    units: int = Field(..., ge=0, le=200, description="Units to order (0–200).")

    @property
    def arguments(self) -> Dict[str, Any]:
        return {"units": self.units}


class StartTaskAction(BaseModel):
    """Convenience model for the start_task tool."""
    tool_name: str = Field(default="start_task", frozen=True)
    task_name: str = Field(
        ...,
        description="Task name: stable_procurement | volatile_demand | crisis_management",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    @property
    def arguments(self) -> Dict[str, Any]:
        return {"task_name": self.task_name, "seed": self.seed}


# ---------------------------------------------------------------------------
# Observation models
# ---------------------------------------------------------------------------

class WarehouseState(BaseModel):
    """Snapshot of the warehouse at a point in time."""

    task: str = Field(default="", description="Active task name.")
    day: int = Field(default=0, description="Current simulation day.")
    max_days: int = Field(default=0, description="Maximum days for this task.")
    stock: float = Field(default=0.0, description="Current inventory level (units).")
    capital: float = Field(default=0.0, description="Available capital ($).")
    storage_capacity: int = Field(default=150, description="Maximum storage (units).")
    supplier_strike: bool = Field(
        default=False,
        description="True when the supplier strike is active — no deliveries possible.",
    )
    done: bool = Field(default=False, description="True when the episode has ended.")
    step_count: int = Field(default=0, description="Total steps taken this episode.")

    model_config = {"extra": "allow"}


class DayResult(BaseModel):
    """Result returned by advance_day()."""

    day: int = Field(..., description="Day number after advancing.")
    delivered: int = Field(default=0, description="Units received from supplier.")
    spoiled: int = Field(default=0, description="Units lost to spoilage.")
    demand: int = Field(..., description="Customer demand for the day.")
    units_sold: int = Field(..., description="Units actually sold.")
    stockout: int = Field(default=0, description="Unmet demand (stockout quantity).")
    revenue: float = Field(..., description="Revenue earned this day ($).")
    day_reward: float = Field(..., description="Reward signal for this day (–0.5 to 1.0).")
    done: bool = Field(default=False, description="True if the episode has ended.")
    supplier_strike_active: bool = Field(
        default=False, description="Strike status after day resolution."
    )

    # Inherits warehouse snapshot fields
    stock: float = Field(default=0.0)
    capital: float = Field(default=0.0)

    model_config = {"extra": "allow"}


class EpisodeScore(BaseModel):
    """Final score returned by get_score()."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised episode score in [0.0, 1.0].",
    )
    capital_ratio: float = Field(..., description="final_capital / initial_capital.")
    avg_day_reward: float = Field(..., description="Mean day reward over episode.")
    episode_complete: bool = Field(
        ..., description="True if all days completed without bankruptcy."
    )
    task_name: str = Field(..., description="Task that was evaluated.")
    days_survived: int = Field(..., description="Number of days completed.")
    max_days: int = Field(..., description="Maximum days for this task.")
    final_capital: float = Field(..., description="Capital at episode end ($).")
    final_stock: float = Field(..., description="Inventory at episode end (units).")


class WarehouseObservation(BaseModel):
    """
    Observation returned after any environment interaction.

    For step results: contains reward, done, and a result payload.
    The payload shape depends on which tool was called:
      - advance_day  → DayResult fields
      - get_score    → EpisodeScore fields
      - others       → WarehouseState fields + optional message
    """

    done: bool = Field(default=False, description="True when the episode has ended.")
    reward: float = Field(default=0.0, description="Cumulative reward so far.")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw tool result payload (varies by tool).",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (status messages, errors, etc.).",
    )

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Task metadata (convenience constants)
# ---------------------------------------------------------------------------

TASK_NAMES = [
    "stable_procurement",
    "volatile_demand",
    "crisis_management",
]

TASK_DIFFICULTIES = {
    "stable_procurement": "easy",
    "volatile_demand": "medium",
    "crisis_management": "hard",
}

__all__ = [
    # Actions
    "WarehouseAction",
    "PlaceOrderAction",
    "StartTaskAction",
    # Observations
    "WarehouseObservation",
    "WarehouseState",
    "DayResult",
    "EpisodeScore",
    # Constants
    "TASK_NAMES",
    "TASK_DIFFICULTIES",
]
