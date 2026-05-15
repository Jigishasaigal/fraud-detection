import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np

from datetime import datetime
from streamlit_autorefresh import st_autorefresh

API = "http://127.0.0.1:8002"

# ---------------- AUTO REFRESH ----------------

st_autorefresh(
    interval=4000,
    key="fraud_stream_refresh"
)

# ---------------- LOAD ANALYTICS DATA ----------------

dashboard_df = pd.read_csv("dashboard_data.csv")

dashboard_df["timestamp"] = pd.to_datetime(
    dashboard_df["timestamp"]
)

# ---------------- LIVE STREAM SIMULATION ----------------

# ---------------- LIVE STREAM SIMULATION ----------------

def generate_live_transaction():

    prob = np.random.beta(0.15, 40)

    if prob < 0.02:

        risk = "LOW RISK"
        decision = "Approve"

    elif prob < 0.08:

        risk = "MEDIUM RISK"
        decision = "Review"

    else:

        risk = "HIGH RISK"
        decision = "Block"

    return {

        "timestamp": datetime.now().strftime("%H:%M:%S"),

        "fraud_probability": round(prob, 6),

        "risk": risk,

        "decision": decision,

        "region": np.random.choice(
            ["North", "South", "East", "West"]
        ),

        "amount": np.random.randint(
            100,
            50000
        )
    }

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fraud Risk Intelligence",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0e1117;
    color: #e6e6e6;
}

h1, h2, h3 {
    color: #ffffff;
}

.metric-box {
    padding: 25px;
    border-radius: 16px;
    background: #161b22;
    border: 1px solid #30363d;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}

.low-risk {
    background: linear-gradient(135deg, #0f5132, #198754);
}

.medium-risk {
    background: linear-gradient(135deg, #664d03, #ffc107);
}

.high-risk {
    background: linear-gradient(135deg, #842029, #dc3545);
}

.trace-step {
    background: #161b22;
    border-left: 4px solid #0d6efd;
    padding: 14px;
    border-radius: 8px;
    margin-bottom: 12px;
}

.small-muted {
    color: #9aa4af;
    font-size: 0.85rem;
}

.analytics-card {
    background: #161b22;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #30363d;
}

.live-alert {
    background: #2b0b0e;
    border-left: 5px solid #dc3545;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("# 💳 Fraud Risk Monitoring & Intelligence Platform")

st.markdown(
    """
    <div class='small-muted'>
    Real-time fraud analytics, explainability, monitoring, and analyst intelligence
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SESSION STATE ----------------

for k in [
    "prob",
    "risk",
    "decision",
    "risk_class",
    "trace",
    "explain",
    "counter",
    "qa_answer"
]:
    if k not in st.session_state:
        st.session_state[k] = None

if "live_transactions" not in st.session_state:
    st.session_state.live_transactions = []

# ---------------- LIVE TRANSACTION UPDATE ----------------

new_tx = generate_live_transaction()

st.session_state.live_transactions.insert(0, new_tx)

# keep latest 20 only
st.session_state.live_transactions = (
    st.session_state.live_transactions[:20]
)

# ---------------- KPI SECTION ----------------

total_tx = len(dashboard_df)

fraud_tx = len(
    dashboard_df[
        dashboard_df["risk_label"] == "HIGH RISK"
    ]
)

fraud_rate = (fraud_tx / total_tx) * 100

avg_risk = dashboard_df["fraud_probability"].mean()

blocked_tx = len(
    dashboard_df[
        dashboard_df["decision"] == "Block"
    ]
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Transactions", f"{total_tx:,}")
c2.metric("Fraud Alerts", fraud_tx)
c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
c4.metric("Avg Risk", f"{avg_risk:.4f}")
c5.metric("Blocked", blocked_tx)

st.divider()

# ---------------- ANALYTICS CHARTS ----------------

left, right = st.columns(2)

# -------- TRANSACTION TREND --------

with left:

    trend_df = (
        dashboard_df
        .groupby(
            dashboard_df["timestamp"].dt.date
        )
        .size()
        .reset_index(name="transactions")
    )

    fig = px.line(
        trend_df,
        x="timestamp",
        y="transactions",
        title="📈 Transaction Volume Trend",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------- RISK DISTRIBUTION --------

with right:

    risk_counts = (
        dashboard_df["risk_label"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ["risk", "count"]

    fig2 = px.pie(
        risk_counts,
        names="risk",
        values="count",
        title="⚠️ Risk Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.divider()

# ---------------- REGION ANALYTICS ----------------

region_counts = (
    dashboard_df
    .groupby("region")
    .size()
    .reset_index(name="transactions")
)

fig3 = px.bar(
    region_counts,
    x="region",
    y="transactions",
    title="🌍 Region-wise Transaction Distribution"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.divider()

# ---------------- SUSPICIOUS TRANSACTIONS ----------------

st.subheader("🚨 Suspicious Transactions")

suspicious_df = dashboard_df.sort_values(
    by="fraud_probability",
    ascending=False
).head(10)

st.dataframe(
    suspicious_df,
    use_container_width=True
)

st.divider()

# ---------------- LIVE FRAUD MONITOR ----------------

st.subheader("⚡ Real-Time Fraud Monitoring")

live_df = pd.DataFrame(
    st.session_state.live_transactions
)

st.dataframe(
    live_df,
    use_container_width=True
)

# ---------------- LIVE HIGH RISK ALERTS ----------------

high_risk_live = live_df[
    live_df["risk"] == "HIGH RISK"
]

if not high_risk_live.empty:

    st.subheader("🚨 Live Fraud Alerts")

    for _, row in high_risk_live.head(5).iterrows():

        st.markdown(
            f"""
            <div class="live-alert">
                <b>HIGH RISK TRANSACTION DETECTED</b><br/>
                Time: {row['timestamp']}<br/>
                Probability: {row['fraud_probability']}<br/>
                Region: {row['region']}<br/>
                Amount: ₹{row['amount']}
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# ---------------- SIDEBAR INPUTS ----------------

with st.sidebar:

    st.markdown("## 🧾 Transaction Evaluation")

    st.markdown(
        """
        <div class='small-muted'>
        Simulate and evaluate a transaction in real time
        </div>
        """,
        unsafe_allow_html=True
    )

    features = [
        st.number_input(
            f"Feature {i+1}",
            value=0.0,
            key=f"f{i}"
        )
        for i in range(30)
    ]

    evaluate = st.button(
        "🔍 Evaluate Transaction",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if evaluate:

    resp = requests.post(
        f"{API}/predict",
        json={"features": features}
    )

    data = resp.json()

    st.session_state.prob = data["fraud_probability"]
    st.session_state.trace = data["decision_trace"]

    if st.session_state.prob < 0.002:

        st.session_state.risk = "LOW RISK"
        st.session_state.decision = "Approve"
        st.session_state.risk_class = "low-risk"

    elif st.session_state.prob < 0.01:

        st.session_state.risk = "MEDIUM RISK"
        st.session_state.decision = "Review"
        st.session_state.risk_class = "medium-risk"

    else:

        st.session_state.risk = "HIGH RISK"
        st.session_state.decision = "Block"
        st.session_state.risk_class = "high-risk"

# ---------------- MAIN RISK OUTPUT ----------------

if st.session_state.prob is not None:

    st.markdown(
        f"""
        <div class="metric-box {st.session_state.risk_class}">
            <h2>{st.session_state.risk}</h2>
            <div>Decision: <b>{st.session_state.decision}</b></div>
            <div class="small-muted">Fraud Probability</div>
            <h2>{st.session_state.prob:.6f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    payload = {
        "fraud_probability": st.session_state.prob,
        "risk_label": st.session_state.risk,
        "top_features": ["V14", "V12", "V4"],
        "decision_trace": st.session_state.trace
    }

    # ---------------- TABS ----------------

    tab1, tab2, tab3 = st.tabs([
        "🧠 Explanation",
        "🔄 Counterfactual",
        "🧑‍💼 Analyst Q&A"
    ])

    # -------- EXPLANATION --------

    with tab1:

        if st.button("Generate Explanation"):

            r = requests.post(
                f"{API}/explain",
                json=payload
            )

            data = r.json()

            st.session_state.explain = data["explanation"]
            st.session_state.trace = data["decision_trace"]

        if st.session_state.explain:

            st.markdown("### Why this decision?")
            st.write(st.session_state.explain)

    # -------- COUNTERFACTUAL --------

    with tab2:

        if st.button("What would change the decision?"):

            r = requests.post(
                f"{API}/explain/counterfactual",
                json=payload
            )

            data = r.json()

            st.session_state.counter = (
                data["counterfactual_explanation"]
            )

            st.session_state.trace = (
                data["decision_trace"]
            )

        if st.session_state.counter:

            st.markdown("### Counterfactual Scenario")
            st.write(st.session_state.counter)

    # -------- ANALYST Q&A --------

    with tab3:

        q = st.text_input(
            "Ask an analyst-style question"
        )

        if st.button("Ask Analyst"):

            r = requests.post(
                f"{API}/explain/qa",
                json=payload | {"question": q}
            )

            data = r.json()

            st.session_state.qa_answer = data["answer"]

            st.session_state.trace = (
                data["decision_trace"]
            )

        if st.session_state.qa_answer:

            st.markdown("### Analyst Answer")
            st.write(st.session_state.qa_answer)

    st.divider()

    # ---------------- DECISION TRACE ----------------

    st.markdown("## 🧾 Decision Trace")

    for step in st.session_state.trace["steps"]:

        st.markdown(
            f"""
            <div class="trace-step">
                <b>{step['step']}</b><br/>
                <span class="small-muted">
                {step['details']}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
