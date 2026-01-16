from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from rag.explain import (
    retrieve_evidence,
    generate_explanation,
    generate_counterfactual_explanation,
    answer_analyst_question,
    confidence_label
)

from utils.decision_trace import init_trace, add_step

app = FastAPI(title="Fraud Detection API")
model = joblib.load("models/fraud_model.pkl")

# ---------------- SCHEMAS ----------------
class Transaction(BaseModel):
    features: List[float]

class ExplainRequest(BaseModel):
    fraud_probability: float
    risk_label: str
    top_features: List[str]
    decision_trace: Optional[Dict[str, Any]] = None

class QARequest(ExplainRequest):
    question: str

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(tx: Transaction):
    trace = init_trace()

    X = np.array(tx.features).reshape(1, -1)
    prob = float(model.predict_proba(X)[0][1])

    if prob < 0.002:
        risk, decision = "LOW RISK", "Approve"
    elif prob < 0.01:
        risk, decision = "MEDIUM RISK", "Review"
    else:
        risk, decision = "HIGH RISK", "Block"

    add_step(trace, "Model Prediction", {"fraud_probability": prob})
    add_step(trace, "Risk & Decision Policy", {
        "risk_label": risk,
        "decision": decision
    })

    trace["steps"] = sorted(trace["steps"], key=lambda x: x["order"])

    return {
        "fraud_probability": prob,
        "decision_trace": trace
    }

# ---------------- EXPLAIN ----------------
@app.post("/explain")
def explain(req: ExplainRequest):
    trace = req.decision_trace or init_trace()

    add_step(trace, "Input Context", {
        "risk_label": req.risk_label,
        "top_features": req.top_features
    })

    evidence = retrieve_evidence(req.fraud_probability, req.risk_label, req.top_features)
    add_step(trace, "Evidence Retrieval", {"num_cases": len(evidence)})

    explanation = generate_explanation(
        req.fraud_probability, req.risk_label, req.top_features, evidence
    )

    add_step(trace, "Explanation Generated", {
        "confidence": confidence_label(req.fraud_probability)
    })

    trace["steps"] = sorted(trace["steps"], key=lambda x: x["order"])

    return {
        "explanation": explanation,
        "decision_trace": trace
    }

# ---------------- COUNTERFACTUAL ----------------
@app.post("/explain/counterfactual")
def counterfactual(req: ExplainRequest):
    trace = req.decision_trace or init_trace()

    text = generate_counterfactual_explanation(
        req.fraud_probability, req.risk_label, req.top_features, []
    )

    add_step(trace, "Counterfactual Analysis", {
        "summary": "Multi-feature anomalies required to change risk classification"
    })

    trace["steps"] = sorted(trace["steps"], key=lambda x: x["order"])

    return {
        "counterfactual_explanation": text,
        "decision_trace": trace
    }

# ---------------- ANALYST Q&A ----------------
@app.post("/explain/qa")
def analyst_qa(req: QARequest):
    trace = req.decision_trace or init_trace()

    add_step(trace, "Analyst Question", {"question": req.question})

    answer = answer_analyst_question(
        req.question,
        req.fraud_probability,
        req.risk_label,
        req.top_features,
        []
    )

    add_step(trace, "Analyst Answer Generated", {"status": "answered"})

    trace["steps"] = sorted(trace["steps"], key=lambda x: x["order"])

    return {
        "answer": answer,
        "decision_trace": trace
    }
