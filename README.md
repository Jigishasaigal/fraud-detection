                                                                                Fraud Risk Intelligence System

An end-to-end fraud detection and explainability system combining classical machine learning, cost-aware decision logic, retrieval-augmented generation (RAG), and a modern analyst dashboard. 
This project focuses not only on accurate fraud prediction, but also on transparent, auditable decision-making, which is critical in real financial systems.

🚀 Key Features :

1. Machine Learning Fraud Detection
Trained a LightGBM model on anonymized transaction features
Handles highly imbalanced fraud data
Outputs calibrated fraud probabilities

2. Risk-Aware Decision Policy
Converts probabilities into LOW / MEDIUM / HIGH risk
Business decisions: Approve / Review / Block
Thresholds optimized using cost-sensitive evaluation

3. Explainability Layer
Evidence retrieval using FAISS + sentence embeddings
LLM-generated explanations grounded in historical cases
Deterministic feature-importance explanations (no hallucinations)

4. Counterfactual Explanations
Explains what would need to change for a different decision
Uses abstract, multi-feature anomaly reasoning

5. Decision Trace (Audit Trail)
Captures every reasoning step:
    Model prediction
    Risk policy
    Evidence retrieval
    Explanation
    Counterfactual analysis
    Analyst Q&A
Fully transparent and reproducible

6. Modern Analyst Dashboard
Built with Streamlit
Risk banner, explanation tabs, decision timeline
Designed for fraud analysts and auditors, not end-user

System Architecture :

Transaction Input
        ↓
ML Fraud Model (LightGBM)
        ↓
Fraud Probability
        ↓
Risk & Decision Policy
        ↓
Evidence Retrieval (FAISS)
        ↓
LLM-based Explanation (RAG)
        ↓
Decision Trace (Audit Log)
        ↓
Analyst Dashboard (Streamlit)

📊 Model Performance :

ROC-AUC: ~0.98
Fraud prevalence: ~0.17% (highly imbalanced)
Optimized decision threshold: 0.01 (cost-sensitive)

📌 Disclaimer:

This project uses anonymized / synthetic transaction features.
Feature names (e.g., V14, V12) are latent and do not represent real-world attributes.
