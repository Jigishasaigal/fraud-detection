# 💳 Fraud Risk Monitoring & Decision Intelligence Platform

An end-to-end fraud analytics and monitoring platform that combines machine learning, retrieval-augmented explainability, analyst workflows, and real-time fraud monitoring into a deployable intelligence system.

The platform goes beyond simple fraud prediction by providing:

* risk-based decision intelligence
* retrieval-grounded explanations
* counterfactual analysis
* analyst Q&A workflows
* audit traceability
* business intelligence dashboards
* real-time fraud monitoring simulation

---

# 🚀 Features

## 🔹 Fraud Risk Prediction

* LightGBM-based fraud classification model
* Handles highly imbalanced transaction data
* Probability-based fraud scoring
* Risk-aware decision policy:

  * LOW RISK → Approve
  * MEDIUM RISK → Review
  * HIGH RISK → Block

---

## 🔹 Retrieval-Augmented Explainability (RAG)

* FAISS-powered evidence retrieval
* Sentence-transformer embeddings for semantic similarity
* LLM-generated explanations grounded in retrieved evidence
* Reduces hallucinated explanations by constraining responses to retrieved cases

---

## 🔹 Counterfactual Reasoning

The system explains:

* what patterns contributed to the current decision
* what would need to change for the transaction risk to change

This provides analyst-friendly transparency for fraud investigations.

---

## 🔹 Analyst Q&A Interface

Supports natural-language analyst questions such as:

* Why does V14 matter?
* Why was this transaction approved?
* What changed the fraud score?

The system uses deterministic routing and evidence-grounded reasoning to avoid unsupported semantic claims about latent features.

---

## 🔹 Decision Trace / Audit Trail

Every prediction generates a structured reasoning timeline:

1. Model prediction
2. Risk policy evaluation
3. Evidence retrieval
4. Explanation generation
5. Counterfactual analysis
6. Analyst interaction tracking

This enables auditability and investigation transparency.

---

# 📊 Business Intelligence & Monitoring Dashboard

The Streamlit dashboard provides:

## ✅ KPI Monitoring

* Total transactions
* Fraud alerts
* Fraud rate
* Average risk score
* Blocked transactions

## ✅ Fraud Analytics

* Transaction trend visualization
* Risk distribution charts
* Region-wise fraud monitoring
* Suspicious transaction tables

## ✅ Real-Time Monitoring Simulation

* Live fraud stream simulation
* Dynamic risk updates
* Real-time fraud alerts
* Streaming transaction monitoring

---

# 🏗️ System Architecture

```text
User Dashboard (Streamlit)
        ↓
FastAPI Backend
        ↓
Fraud Prediction Model (LightGBM)
        ↓
Risk Decision Engine
        ↓
RAG Explainability Layer
   ├── FAISS Retrieval
   ├── Sentence Embeddings
   └── LLM Explanations (Groq / LLaMA 3.1)
```

---

# 🧠 Machine Learning Pipeline

## Model

* LightGBM classifier

## Evaluation

* ROC-AUC based evaluation
* Cost-sensitive threshold optimization
* Imbalanced fraud detection handling

## Explainability

* SHAP-based offline feature contribution analysis
* Retrieval-grounded natural language explanations

---

# 🧰 Tech Stack

## Machine Learning

* LightGBM
* Scikit-learn
* NumPy
* Pandas

## Explainability & RAG

* FAISS
* Sentence-Transformers
* SHAP

## Backend

* FastAPI
* Uvicorn
* Pydantic

## LLM Integration

* Groq API
* LLaMA 3.1

## Frontend

* Streamlit
* Plotly

## Deployment

* Render
* GitHub

---

# 📂 Project Structure

```text
fraud-detection/
│
├── app.py
├── dashboard.py
├── dashboard_data.csv
├── requirements.txt
│
├── models/
│   └── fraud_model.pkl
│
├── rag/
│   └── explain.py
│
├── utils/
│   └── decision_trace.py
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── threshold_optimization.py
│   ├── shap_explainability.py
│   ├── compute_business_impact.py
│   ├── business_metrics.py
│   └── generate_dashboard_data.py
│
└── data/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/Jigishasaigal/fraud-detection.git
cd fraud-detection
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

# ▶️ Running the Project

## Start FastAPI Backend

```bash
python -m uvicorn app:app --port 8002
```

---

## Start Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

# 🌐 API Endpoints

| Endpoint                  | Description                  |
| ------------------------- | ---------------------------- |
| `/predict`                | Fraud probability prediction |
| `/explain`                | Risk explanation generation  |
| `/explain/counterfactual` | Counterfactual reasoning     |
| `/explain/qa`             | Analyst Q&A                  |

---

# 📈 Example Dashboard Capabilities

* Fraud trend monitoring
* Region-wise analytics
* Real-time fraud alerts
* Decision intelligence
* Investigation workflows
* Evidence-grounded explanations

---

# 📌 Key Highlights

* End-to-end fraud intelligence workflow
* Retrieval-augmented explainability system
* Real-time monitoring simulation
* Analyst-focused decision support
* Explainable AI + BI integration
* Deployable architecture with FastAPI and Streamlit

---

# 📄 Resume Summary

Built a Fraud Risk Monitoring & Decision Intelligence Platform using LightGBM, FastAPI, FAISS, and LLM-powered explainability. Integrated real-time fraud monitoring, retrieval-grounded explanations, analyst Q&A workflows, and business intelligence dashboards into a deployable end-to-end system.

---

# ⚠️ Notes

* The dashboard uses synthetic transaction simulation for monitoring demonstrations.
* Raw datasets are excluded from the repository due to size constraints.
* The project is intended for educational, portfolio, and research purposes.

---

# 👩‍💻 Author

Jigisha Saigal
