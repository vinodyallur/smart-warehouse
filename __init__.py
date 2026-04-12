"""
Supply Chain Digital Twin — top-level package.

Quick start:
    from supply_chain_env import SupplyChainEnv, WarehouseAction, WarehouseObservation

    with SupplyChainEnv(base_url="http://localhost:8000") as env:
        env.reset()
        env.call_tool("start_task", task_name="stable_procurement", seed=42)
        env.call_tool("place_order", units=30)
        result = env.call_tool("advance_day")
        score  = env.call_tool("get_score")
"""

# MCP action types (used by the HTTP client internally)
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

# Client
from .client import SupplyChainEnv

# Typed models (required by openenv validate)
from .models import (
    WarehouseAction,
    PlaceOrderAction,
    StartTaskAction,
    WarehouseObservation,
    WarehouseState,
    DayResult,
    EpisodeScore,
    TASK_NAMES,
    TASK_DIFFICULTIES,
)

__all__ = [
    # Client
    "SupplyChainEnv",
    # MCP types
    "CallToolAction",
    "ListToolsAction",
    # Domain models
    "WarehouseAction",
    "PlaceOrderAction",
    "StartTaskAction",
    "WarehouseObservation",
    "WarehouseState",
    "DayResult",
    "EpisodeScore",
    # Constants
    "TASK_NAMES",
    "TASK_DIFFICULTIES",
]
