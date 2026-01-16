from datetime import datetime

STEP_ORDER = {
    "Model Prediction": 1,
    "Risk & Decision Policy": 2,
    "Input Context": 3,
    "Evidence Retrieval": 4,
    "Explanation Generated": 5,
    "Counterfactual Analysis": 6,
    "Analyst Question": 7,
    "Analyst Answer Generated": 8,
}

def init_trace():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "steps": []
    }

def add_step(trace, name, details):
    trace["steps"].append({
        "order": STEP_ORDER.get(name, 99),
        "step": name,
        "details": details
    })
