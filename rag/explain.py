

from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import faiss

from sentence_transformers import (
    SentenceTransformer
)

from groq import Groq

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "index.faiss"
)

META_PATH = os.path.join(
    BASE_DIR,
    "index_meta.pkl"
)

# =========================
# LAZY LOADED ASSETS
# =========================

_embedder = None
_index = None
_metadata = None

# =========================
# GROQ CLIENT
# =========================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# LOAD RAG ASSETS
# =========================

def _load_rag_assets():

    global _embedder
    global _index
    global _metadata

    if _embedder is None:

        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    if _index is None:

        _index = faiss.read_index(
            INDEX_PATH
        )

    if _metadata is None:

        with open(META_PATH, "rb") as f:

            _metadata = pickle.load(f)

# =========================
# RETRIEVE EVIDENCE
# =========================

def retrieve_evidence(
    prob,
    risk,
    features,
    k=4
):

    _load_rag_assets()

    query = (
        f"Fraud probability {prob}, "
        f"risk {risk}, "
        f"features {', '.join(features)}"
    )

    emb = _embedder.encode([query])

    _, idxs = _index.search(
        emb,
        k
    )

    evidence = [
        _metadata[i]
        for i in idxs[0]
    ]

    if risk == "LOW RISK":

        legit = [

            e for e in evidence

            if (
                "Legitimate" in e["text"]
                or
                "Approved" in e["text"]
            )
        ]

        return legit[:2] if legit else evidence[:1]

    return evidence

# =========================
# CONFIDENCE LABEL
# =========================

def confidence_label(prob):

    if prob < 0.02:
        return "HIGH CONFIDENCE"

    elif prob < 0.08:
        return "MODERATE CONFIDENCE"

    return "HIGH CONFIDENCE"

# =========================
# MAIN EXPLANATION
# =========================

def generate_explanation(
    prob,
    risk,
    features,
    evidence
):

    ev_text = "\n".join(

        f"- ({e['source']}) {e['text']}"

        for e in evidence
    )

    if risk == "LOW RISK":

        instruction = """
Explain why this transaction is LOW RISK.
Focus on absence of known fraud patterns.
Do not speculate.
"""

    else:

        instruction = """
Explain why this transaction is HIGH RISK.
Focus on similarity to anomalous fraud patterns.
"""

    confidence = confidence_label(prob)

    prompt = f"""
You are a financial risk analyst.

Confidence level:
{confidence}

{instruction}

Rules:
- Use ONLY provided evidence
- Do NOT speculate
- Do NOT assign semantic meanings to latent features
- Avoid exaggerated certainty
- Use cautious analyst-style language

Fraud probability:
{prob}

Risk level:
{risk}

Key features:
{', '.join(features)}

Evidence:
{ev_text}

Write 3-5 sentences.
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content":
                "Explain fraud decisions using retrieved evidence only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1
    )

    return (
        response
        .choices[0]
        .message
        .content
    )

# =========================
# COUNTERFACTUAL
# =========================

def generate_counterfactual_explanation(
    prob,
    risk,
    features,
    evidence
):

    prompt = f"""
You are a financial risk analyst.

Task:
Explain abstractly what would need to change
for this transaction to become HIGH RISK.

STRICT RULES:
- Use abstract language only
- No semantics for features
- No thresholds
- No numeric movement
- No speculation
- Mention only anomalous multi-feature behavior

Features:
{', '.join(features)}

Write exactly 1 sentence.
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content":
                "Generate minimal counterfactual explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.0
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

# =========================
# ANALYST QA
# =========================

def answer_analyst_question(
    question,
    prob,
    risk,
    features,
    evidence
):

    ev_text = "\n".join(

        f"- ({e['source']}) {e['text']}"

        for e in evidence
    )

    prompt = f"""
You are a fraud analyst assistant.

Question:
{question}

Guidelines:
- Explain feature importance at model level
- Avoid assigning semantic meaning
- Do not speculate
- Do not say importance is unknown
- Use cautious analyst language

Transaction context:
Risk level: {risk}

Features:
{', '.join(features)}

Evidence:
{ev_text}

Write 2-3 concise sentences.
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content":
                "You answer analyst questions conservatively."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.0
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )
