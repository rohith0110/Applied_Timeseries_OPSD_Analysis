"""
Streamlit Dashboard for Live Simulation.
Run with: streamlit run src/online/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import datetime


st.set_page_config(page_title="GridGuard Live Dashboard", layout="wide")


st.sidebar.title("Configuration")
country = st.sidebar.selectbox("Select Country", ["DK", "ES", "FR"], index=0)


@st.cache_data(ttl=60)
def load_data(country_code):
    results_path = Path(f"outputs/{country_code}_online_results.csv")
    logs_path = Path(f"outputs/{country_code}_online_updates.csv")

    if not results_path.exists():
        return None, None

    res_df = pd.read_csv(results_path, parse_dates=["timestamp"])
    log_df = pd.read_csv(logs_path, parse_dates=["timestamp"])
    return res_df, log_df


res_df, log_df = load_data(country)

if res_df is None:
    st.error(f"No data found for {country}. Please run the simulation first.")
    st.stop()


st.title(f"⚡ GridGuard Live Monitor: {country}")


last_ts = res_df["timestamp"].iloc[-1]

seven_days_ago = last_ts - pd.Timedelta(days=7)
recent_df = res_df[res_df["timestamp"] > seven_days_ago]


y = recent_df["load"].values
if len(y) > 24:
    naive_err = abs(y[24:] - y[:-24]).mean() + 1e-8
    mae = abs(recent_df["load"] - recent_df["yhat"]).mean()
    mase = mae / naive_err
else:
    mase = 0.0


covered = (recent_df["load"] >= recent_df["yhat_lo"]) & (
    recent_df["load"] <= recent_df["yhat_hi"]
)
coverage = covered.mean() * 100


today_start = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
today_df = res_df[res_df["timestamp"] >= today_start]
anomalies_today = today_df["is_anomaly"].sum()


if not log_df.empty:
    last_update = log_df.iloc[-1]
    last_update_time = last_update["timestamp"]
    last_update_reason = last_update["reason"]
else:
    last_update_time = "N/A"
    last_update_reason = "N/A"


col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Rolling 7d MASE",
    f"{mase:.3f}",
    help="Mean Absolute Scaled Error over the last 7 days of simulation",
)
col2.metric(
    "80% PI Coverage",
    f"{coverage:.1f}%",
    help="Percentage of actuals within the 80% prediction interval (last 7 days)",
)
col3.metric(
    "Anomalies Today",
    f"{anomalies_today} hrs",
    help="Number of anomalies detected since 00:00 today",
)
col4.metric(
    "Last Update",
    f"{last_update_time}",
    delta=last_update_reason,
    help="Timestamp of the last model adaptation",
)


st.subheader("Live Load Series & Forecast")


days_to_show = st.slider("History (Days)", 1, 14, 7)
plot_start = last_ts - pd.Timedelta(days=days_to_show)
plot_df = res_df[res_df["timestamp"] > plot_start]

fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["load"],
        name="Actual Load",
        line=dict(color="#FFFFFF", width=2),  # White for dark mode visibility
    )
)


fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["yhat"],
        name="Forecast (Median)",
        line=dict(color="#00B5F0", width=2),  # Cyan/Bright Blue
    )
)


fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["yhat_hi"],
        name="80% PI Upper",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["yhat_lo"],
        name="80% PI",
        fill="tonexty",
        fillcolor="rgba(0, 181, 240, 0.2)",
        line=dict(width=0),
        hoverinfo="skip",
    )
)


anoms = plot_df[plot_df["is_anomaly"]]
if not anoms.empty:
    fig.add_trace(
        go.Scatter(
            x=anoms["timestamp"],
            y=anoms["load"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=8, symbol="x"),
        )
    )


plot_logs = log_df[log_df["timestamp"] > plot_start]
for _, row in plot_logs.iterrows():
    color = "green" if row["reason"] == "drift" else "gray"
    fig.add_vline(
        x=row["timestamp"].timestamp() * 1000,
        line_width=1,
        line_dash="dash",
        line_color=color,
    )

fig.update_layout(height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)


col_tape, col_drift = st.columns([1, 1])

with col_tape:
    st.subheader("Anomaly Tape (Last 24h)")

    tape_df = res_df.iloc[-24:].copy()
    tape_df["hour"] = tape_df["timestamp"].dt.hour

    fig_tape = go.Figure(
        data=go.Heatmap(
            z=[tape_df["z_score"].abs()],
            x=tape_df["timestamp"],
            y=["Z-Score"],
            colorscale="Reds",
            zmin=0,
            zmax=5,
        )
    )
    fig_tape.update_layout(
        height=150, yaxis_visible=False, margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_tape, use_container_width=True)

with col_drift:
    st.subheader("Drift Monitor")

    fig_drift = go.Figure()
    fig_drift.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["drift_ewma"],
            name="Drift EWMA",
            line=dict(color="#AB63FA"),  # Purple
        )
    )

    fig_drift.add_hline(
        y=3.0,
        line_dash="dot",
        line_color="#FFA15A",
        annotation_text="Threshold",
    )

    fig_drift.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_drift, use_container_width=True)


st.subheader("Recent Adaptation Events")
st.dataframe(log_df.tail(10).sort_values("timestamp", ascending=False))
