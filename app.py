from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from utils.decision_trace import (
    init_trace,
    add_step
)

app = FastAPI(
    title="Fraud Detection API"
)

# ---------------- MODEL ----------------

model = joblib.load(
    "models/fraud_model.pkl"
)

# ---------------- REQUEST SCHEMAS ----------------

class Transaction(BaseModel):
    features: List[float]

class ExplainRequest(BaseModel):
    fraud_probability: float
    risk_label: str
    top_features: List[str]

class QARequest(BaseModel):
    question: str
    fraud_probability: float
    risk_label: str
    top_features: List[str]

# =========================
# PREDICT ENDPOINT
# =========================

@app.post("/predict")
def predict(tx: Transaction):

    trace = init_trace()

    X = np.array(
        tx.features
    ).reshape(1, -1)

    prob = model.predict_proba(X)[0][1]

    add_step(
        trace,
        "Model Prediction",
        {
            "fraud_probability": float(prob)
        }
    )

    if prob < 0.02:

        risk = "LOW RISK"
        decision = "Approve"

    elif prob < 0.08:

        risk = "MEDIUM RISK"
        decision = "Review"

    else:

        risk = "HIGH RISK"
        decision = "Block"

    add_step(
        trace,
        "Risk & Decision Policy",
        {
            "risk_label": risk,
            "decision": decision
        }
    )

    return {
        "fraud_probability": float(prob),
        "decision_trace": trace
    }

# =========================
# EXPLAIN ENDPOINT
# =========================

@app.post("/explain")
def explain(req: ExplainRequest):

    from rag.explain import (
        retrieve_evidence,
        generate_explanation,
        confidence_label
    )

    trace = init_trace()

    add_step(
        trace,
        "Input Context",
        {
            "risk_label": req.risk_label,
            "top_features": req.top_features
        }
    )

    evidence = retrieve_evidence(
        req.fraud_probability,
        req.risk_label,
        req.top_features
    )

    add_step(
        trace,
        "Evidence Retrieval",
        {
            "num_cases": len(evidence)
        }
    )

    text = generate_explanation(
        req.fraud_probability,
        req.risk_label,
        req.top_features,
        evidence
    )

    add_step(
        trace,
        "Explanation Generated",
        {
            "confidence": confidence_label(
                req.fraud_probability
            )
        }
    )

    return {
        "explanation": text,
        "decision_trace": trace
    }

# =========================
# COUNTERFACTUAL ENDPOINT
# =========================

@app.post("/explain/counterfactual")
def counterfactual(req: ExplainRequest):

    from rag.explain import (
        retrieve_evidence,
        generate_counterfactual_explanation
    )

    trace = init_trace()

    evidence = retrieve_evidence(
        req.fraud_probability,
        req.risk_label,
        req.top_features
    )

    text = generate_counterfactual_explanation(
        req.fraud_probability,
        req.risk_label,
        req.top_features,
        evidence
    )

    add_step(
        trace,
        "Counterfactual Analysis",
        {
            "summary":
            "Multi-feature anomalies required to change risk classification"
        }
    )

    return {
        "counterfactual_explanation": text,
        "decision_trace": trace
    }

# =========================
# ANALYST QA ENDPOINT
# =========================

@app.post("/explain/qa")
def analyst_qa(req: QARequest):

    from rag.explain import (
        retrieve_evidence,
        answer_analyst_question
    )

    trace = init_trace()

    add_step(
        trace,
        "Analyst Question",
        {
            "question": req.question
        }
    )

    evidence = retrieve_evidence(
        req.fraud_probability,
        req.risk_label,
        req.top_features
    )

    answer = answer_analyst_question(
        req.question,
        req.fraud_probability,
        req.risk_label,
        req.top_features,
        evidence
    )

    add_step(
        trace,
        "Analyst Answer Generated",
        {
            "status": "answered"
        }
    )

    return {
        "answer": answer,
        "decision_trace": trace
    }
