"""
Supply Chain Digital Twin Client.

Wraps MCPToolClient for convenient interaction with the environment server.

Example:
    >>> with SupplyChainEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     env.call_tool("start_task", task_name="stable_procurement", seed=42)
    ...     env.call_tool("place_order", units=30)
    ...     env.call_tool("advance_day")
    ...     result = env.call_tool("get_score")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class SupplyChainEnv(MCPToolClient):
    """Client for the Supply Chain Digital Twin environment."""
    pass
