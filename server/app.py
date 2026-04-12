"""
FastAPI application for the Supply Chain Digital Twin Environment.

Exposes the standard OpenEnv HTTP API:
  GET  /health         → 200 {"status": "ok"}
  GET  /healthz        → 200 {"status": "ok"}  (alternate)
  GET  /ping           → 200 {"status": "ok"}  (alternate)
  POST /reset          → 200 initial observation
  POST /step           → 200 step observation
  GET  /state          → 200 current state
  GET  /web            → web UI (from openenv-core)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
    uv run --project . server
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .supply_chain_environment import SupplyChainEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.supply_chain_environment import SupplyChainEnvironment

# Pass CLASS (not instance) so each WebSocket session gets its own environment.
app = create_app(
    SupplyChainEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="supply_chain_env",
)

# ------------------------------------------------------------------
# Belt-and-suspenders health routes so the HF Space ping always 200s
# ------------------------------------------------------------------
from fastapi.responses import JSONResponse


@app.get("/healthz", include_in_schema=False)
@app.get("/ping", include_in_schema=False)
async def _ping():
    return JSONResponse({"status": "ok", "env": "supply_chain_env"})


# /health is already registered by create_app, but add a fallback in case
# the openenv-core version running on HF doesn't include it.
try:
    @app.get("/health", include_in_schema=False)
    async def _health():
        return JSONResponse({"status": "ok"})
except Exception:
    pass  # already registered — fine


def main():
    """Entry point: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
