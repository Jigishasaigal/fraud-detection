from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "index_meta.pkl")

# ---------------- LOAD INDEX ----------------
index = faiss.read_index(INDEX_PATH)
metadata = pickle.load(open(META_PATH, "rb"))

# ---------------- MODELS ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- RETRIEVAL ----------------
def retrieve_evidence(prob, risk, features, k=4):
    query = f"Fraud probability {prob}, risk {risk}, features {', '.join(features)}"
    emb = embedder.encode([query])
    _, idxs = index.search(emb, k)

    evidence = [metadata[i] for i in idxs[0]]

    if risk == "LOW RISK":
        legit = [
            e for e in evidence
            if "Legitimate" in e["text"] or "Approved" in e["text"]
        ]
        return legit[:2] if legit else evidence[:1]

    return evidence

# ---------------- CONFIDENCE ----------------
def confidence_label(prob: float) -> str:
    """
    Communication-level confidence (NOT a decision threshold)
    """
    if prob < 0.002:
        return "HIGH CONFIDENCE"
    elif prob < 0.01:
        return "MODERATE CONFIDENCE"
    else:
        return "HIGH CONFIDENCE"

# ---------------- MAIN EXPLANATION ----------------
def generate_explanation(prob, risk, features, evidence):
    ev_text = "\n".join(f"- ({e['source']}) {e['text']}" for e in evidence)

    if risk == "LOW RISK":
        instruction = """
Explain why this transaction is LOW RISK.
Focus on the absence of known fraud patterns.
Do NOT describe fraud cases in detail.
Do NOT assume the presence of anomalies unless explicitly stated.
"""
    else:
        instruction = """
Explain why this transaction is HIGH RISK.
Focus on similarity to confirmed fraud cases.
"""

    confidence = confidence_label(prob)

    prompt = f"""
You are a financial risk analyst.

Confidence level: {confidence}

Guidelines:
- If confidence is HIGH, be clear and decisive.
- If confidence is MODERATE, acknowledge uncertainty and suggest review.
- Do NOT exaggerate risk.
- Do NOT speculate.

{instruction}

Rules:
- Use ONLY the provided evidence.
- Do NOT speculate.

Fraud probability: {prob}
Risk level: {risk}
Key features: {', '.join(features)}

Evidence:
{ev_text}

Write 3–5 sentences in plain English.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Explain fraud decisions using evidence only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

# ---------------- COUNTERFACTUAL ----------------
def generate_counterfactual_explanation(prob, risk, features, evidence):
    prompt = f"""
You are a financial risk analyst.

Task:
Explain, in abstract terms, what would need to change for this transaction to become HIGH RISK.

STRICT RULES:
- Use ONLY abstract language.
- Do NOT assign any real-world or domain meaning to features.
- Do NOT mention case IDs, examples, or scenarios.
- Do NOT speculate about intent, origin, identity, or behavior.
- Do NOT describe numeric movement or thresholds.
- State ONLY that multiple features would exhibit anomalies.

Features:
{', '.join(features)}

Write exactly 1 sentence.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You generate minimal, abstract counterfactual explanations with no semantics."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

def answer_analyst_question(question, prob, risk, features, evidence):
    q = question.lower().strip()

    # 1) Deterministic handling: "Why does <feature> matter?"
    if q.startswith("why does") and "matter" in q:
        feature = (
            q.replace("why does", "")
             .replace("matter", "")
             .replace("?", "")
             .strip()
             .upper()
        )

        if feature in features:
            return (
                f"{feature} matters because the model has learned that anomalous "
                f"behavior in this feature contributes to distinguishing fraudulent "
                f"and legitimate transactions when evaluated alongside other features. "
                f"In this transaction, {feature} does not exhibit anomalous behavior, "
                f"which supports the {risk.lower()} classification."
            )

        return (
            "The feature referenced in the question is not part of the model’s "
            "key feature set for this transaction."
        )

    # 2) Hard refusal: semantic questions
    semantic_terms = [
        "identity", "payment", "customer", "merchant",
        "user", "device", "location", "card"
    ]

    if any(term in q for term in semantic_terms):
        return (
            "The evidence does not assign real-world or domain-specific meaning "
            "to latent feature names, so this question cannot be answered."
        )

    # 3) Fallback
    return (
        "The available evidence does not provide sufficient information "
        "to answer this question."
    )
