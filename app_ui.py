"""
Supply Chain Digital Twin — Command Center Dashboard
Streamlit UI for visualising the AI agent's warehouse decisions in real-time.

Run:  streamlit run app_ui.py
"""

import json
import os
import time
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from openenv.core.mcp_client import MCPToolClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASK_META = {
    "stable_procurement": {"label": "🟢 Easy — Stable Demand", "days": 7},
    "volatile_demand":    {"label": "🟡 Medium — Volatile Demand + Spoilage", "days": 10},
    "crisis_management":  {"label": "🔴 Hard — Supply Crisis", "days": 14},
}

SYSTEM_PROMPT = """You are an expert Chief Supply Chain Officer managing a warehouse.
Your goal is to maximise revenue while keeping inventory healthy and capital positive.
Storage capacity is 150 units. Stockouts and overstock both carry penalties.
On supplier strike days, ordering is impossible.
Before each order provide a 1-sentence Strategic Reasoning explaining your choice."""

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Digital Twin",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Supply Chain Digital Twin — Command Center")
st.caption("AI-powered warehouse management with real-time decision visualisation")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    task_key = st.selectbox(
        "Select Task",
        list(TASK_META.keys()),
        format_func=lambda k: TASK_META[k]["label"],
    )
    seed = st.number_input("Random Seed", value=42, min_value=0)
    run_btn = st.button("▶️  Run Agent", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Environment:** `supply_chain_env`")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown(f"**Endpoint:** `{ENV_BASE_URL}`")

# ---------------------------------------------------------------------------
# Placeholder layout
# ---------------------------------------------------------------------------
kpi_cols = st.columns(4)
cap_ph    = kpi_cols[0].empty()
stock_ph  = kpi_cols[1].empty()
day_ph    = kpi_cols[2].empty()
score_ph  = kpi_cols[3].empty()

strike_ph = st.empty()

col_left, col_right = st.columns([3, 2])
with col_left:
    st.subheader("📈 Capital & Inventory Over Time")
    chart_ph = st.empty()

with col_right:
    st.subheader("🤖 AI Decision Feed")
    feed_ph = st.empty()

gauge_ph = st.empty()


def render_kpis(capital, stock, day, max_days, score, strike):
    cap_ph.metric("💰 Capital", f"${capital:,.0f}")
    stock_ph.metric("📦 Stock", f"{stock:.0f} u")
    day_ph.metric("📅 Day", f"{day} / {max_days}")
    score_ph.metric("🏆 Score", f"{score:.3f}")
    if strike:
        strike_ph.error("🚨 **SUPPLIER STRIKE ACTIVE** — No deliveries today!", icon="⛔")
    else:
        strike_ph.empty()


def render_chart(history: list[dict]):
    if not history:
        return
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["capital"], name="Capital ($)",
                             line=dict(color="#00cc88", width=2)))
    fig.add_trace(go.Scatter(x=df["day"], y=df["stock"] * 10, name="Stock (×10)",
                             line=dict(color="#4488ff", width=2, dash="dot")))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.1),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    chart_ph.plotly_chart(fig, use_container_width=True)


def render_gauge(stock, capacity=150):
    pct = stock / capacity
    color = "#00cc88" if pct < 0.7 else ("#ffaa00" if pct < 0.9 else "#ff4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=stock,
        title={"text": "Inventory Level"},
        gauge={
            "axis": {"range": [0, capacity]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.5 * capacity], "color": "#1a1a2e"},
                {"range": [0.5 * capacity, 0.85 * capacity], "color": "#16213e"},
                {"range": [0.85 * capacity, capacity], "color": "#0f3460"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": 0.85 * capacity},
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    gauge_ph.plotly_chart(fig, use_container_width=True)


def render_feed(feed_lines: list[str]):
    feed_ph.text_area("", "\n\n".join(feed_lines[-12:]), height=300, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
if run_btn:
    feed_lines: list[str] = ["Starting agent…"]
    history: list[dict] = []
    score = 0.0

    render_kpis(0, 0, 0, TASK_META[task_key]["days"], 0.0, False)

    try:
        llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

        with MCPToolClient(base_url=ENV_BASE_URL) as env:
            env.reset()
            state = env.call_tool("start_task", task_name=task_key, seed=int(seed))

            capital  = state.get("capital", 0)
            stock    = state.get("stock", 0)
            strike   = state.get("supplier_strike", False)
            max_days = TASK_META[task_key]["days"]

            render_kpis(capital, stock, 0, max_days, 0.0, strike)
            render_gauge(stock)

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
                        f"Task '{task_key}' started:\n{json.dumps(state, indent=2)}\n\n"
                        "Place the first order, then advance_day. Repeat until done=true."
                    ),
                },
            ]

            done = False
            step = 0
            max_steps = 80

            while step < max_steps and not done:
                try:
                    resp = llm.chat.completions.createPayment(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        max_tokens=512,
                        temperature=0.3,
                    )
                    msg = resp.choices[0].message
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": msg.tool_calls,
                    })

                    if msg.content:
                        feed_lines.append(f"🤖 {msg.content.strip()}")
                        render_feed(feed_lines)

                    if not msg.tool_calls:
                        messages.append({
                            "role": "user",
                            "content": "Call a tool: place_order(units=N) or advance_day().",
                        })
                        continue

                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {}

                        step += 1
                        result = env.call_tool(tool_name, **args)

                        if isinstance(result, dict):
                            capital = result.get("capital", capital)
                            stock   = result.get("stock", stock)
                            strike  = result.get("supplier_strike", strike)
                            day     = result.get("day", 0)
                            done    = result.get("done", False)

                            if "day_reward" in result:
                                history.append({"day": day, "capital": capital, "stock": stock,
                                                "reward": result["day_reward"]})

                            if tool_name == "place_order":
                                units = result.get("units_ordered", args.get("units", 0))
                                status = result.get("status", "")
                                line = f"📦 Day {day} — Order: {units} units ({status})"
                                if strike:
                                    line += " [STRIKE BLOCKED]"
                                feed_lines.append(line)
                            elif tool_name == "advance_day":
                                feed_lines.append(
                                    f"⏩ Day {day}: demand={result.get('demand',0)}, "
                                    f"sold={result.get('units_sold',0)}, "
                                    f"spoiled={result.get('spoiled',0)}, "
                                    f"reward={result.get('day_reward',0):.3f}"
                                )

                        render_kpis(capital, stock, day, max_days, score, strike)
                        render_chart(history)
                        render_gauge(stock)
                        render_feed(feed_lines)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        })

                        if done:
                            break

                        time.sleep(0.05)

                except Exception as exc:
                    feed_lines.append(f"❌ Error: {exc}")
                    render_feed(feed_lines)
                    break

            # Final score
            try:
                sc = env.call_tool("get_score")
                if isinstance(sc, dict):
                    score = sc.get("score", 0.0)
                    feed_lines.append(
                        f"\n✅ Episode complete! Score: {score:.4f} | "
                        f"Capital: ${sc.get('final_capital',0):,.0f} | "
                        f"Days survived: {sc.get('days_survived',0)}/{max_days}"
                    )
            except Exception:
                pass

            render_kpis(capital, stock, day, max_days, score, False)
            render_feed(feed_lines)
            st.success(f"🏆 Final Score: **{score:.4f}**")

    except Exception as exc:
        st.error(f"Connection error: {exc}")
        st.info(f"Make sure the environment server is running at `{ENV_BASE_URL}`")
