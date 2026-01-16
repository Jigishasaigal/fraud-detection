import streamlit as st
import requests

API = "http://127.0.0.1:8002"

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
    padding: 20px;
    border-radius: 12px;
    background: #161b22;
    border: 1px solid #30363d;
    text-align: center;
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
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 10px;
}
.small-muted {
    color: #9aa4af;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("## 💳 Fraud Risk Intelligence Dashboard")
st.markdown(
    "<div class='small-muted'>Real-time transaction risk assessment with explainability and audit trace</div>",
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

# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.markdown("### 🧾 Transaction Features")
    st.markdown(
        "<div class='small-muted'>Latent model features (30)</div>",
        unsafe_allow_html=True
    )

    features = [
        st.number_input(f"Feature {i+1}", value=0.0, key=f"f{i}")
        for i in range(30)
    ]

    evaluate = st.button("🔍 Evaluate Transaction", use_container_width=True)

# ---------------- PREDICTION ----------------
if evaluate:
    resp = requests.post(f"{API}/predict", json={"features": features})
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

# ---------------- MAIN CONTENT ----------------
if st.session_state.prob is not None:

    # -------- RISK BANNER --------
    st.markdown(
        f"""
        <div class="metric-box {st.session_state.risk_class}">
            <h2>{st.session_state.risk}</h2>
            <div>Decision: <b>{st.session_state.decision}</b></div>
            <div class="small-muted">Fraud Probability</div>
            <h3>{st.session_state.prob:.6f}</h3>
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

    # -------- EXPLANATION TABS --------
    tab1, tab2, tab3 = st.tabs(
        ["🧠 Explanation", "🔄 Counterfactual", "🧑‍💼 Analyst Q&A"]
    )

    with tab1:
        if st.button("Generate Explanation"):
            r = requests.post(f"{API}/explain", json=payload)
            data = r.json()
            st.session_state.explain = data["explanation"]
            st.session_state.trace = data["decision_trace"]

        if st.session_state.explain:
            st.markdown("### Why this decision?")
            st.write(st.session_state.explain)

    with tab2:
        if st.button("What would change the decision?"):
            r = requests.post(f"{API}/explain/counterfactual", json=payload)
            data = r.json()
            st.session_state.counter = data["counterfactual_explanation"]
            st.session_state.trace = data["decision_trace"]

        if st.session_state.counter:
            st.markdown("### Counterfactual Scenario")
            st.write(st.session_state.counter)

    with tab3:
        q = st.text_input("Ask an analyst-style question")

        if st.button("Ask Analyst"):
            r = requests.post(
                f"{API}/explain/qa",
                json=payload | {"question": q}
            )
            data = r.json()
            st.session_state.qa_answer = data["answer"]
            st.session_state.trace = data["decision_trace"]

        if st.session_state.qa_answer:
            st.markdown("### Analyst Answer")
            st.write(st.session_state.qa_answer)

    st.divider()

    # -------- DECISION TRACE --------
    st.markdown("## 🧾 Decision Trace")

    for step in st.session_state.trace["steps"]:
        st.markdown(
            f"""
            <div class="trace-step">
                <b>{step['step']}</b><br/>
                <span class="small-muted">{step['details']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
