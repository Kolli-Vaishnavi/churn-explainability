# 🔍 ChurnXplain — Explainable AI Dashboard for Customer Churn

> **Can we trust a machine learning model we can't explain?**
> ChurnXplain goes beyond accuracy — it explains *why* predictions happen,
> compares two explainability methods, and audits for demographic fairness.

---

## 🎯 Problem Statement

Most churn prediction models are black boxes. A business deploying such a model
faces a critical question: *why is this customer predicted to churn?*
Without an answer, retention teams can't act meaningfully — they can only guess.

ChurnXplain solves this by combining:
- **Prediction** — XGBoost churn classifier (ROC-AUC: 0.8346)
- **Explanation** — SHAP and LIME local + global explanations
- **Comparison** — divergence analysis between the two methods
- **Fairness** — demographic parity audit across gender and senior status

---

## 🔬 Research Question

> *When SHAP and LIME disagree on feature importance for the same prediction,
> what does that tell us about model confidence and decision boundary proximity?*

### Key Findings

**SHAP vs LIME Divergence:**
- Both methods agreed on `tenure` and `Contract` as the top predictors of churn
- Disagreement was most common on mid-tier features like `MonthlyCharges`
  vs `TotalCharges`
- High divergence (score ≥ 3/5) correlated with predictions near the 0.4–0.6
  probability range — confirming that divergence signals decision boundary proximity
- **Recommendation:** Predictions with divergence ≥ 3 should be escalated
  for human review rather than automated

**Fairness Audit:**
- Demographic Parity (Gender): 0.0301 — low disparity
- Demographic Parity (Senior Citizen): 0.2309 — Senior citizens predicted
  to churn at a higher rate
- Equalized Odds (Senior): 0.1722
- Senior citizens were predicted to churn at a 14.2% higher rate than non-seniors,
  the largest fairness gap identified. Gender showed negligible disparity (0.032).
  Recommend monitoring senior cohort predictions before production deployment.

---

## 🏗️ Architecture
churn-explainability/

├── data/

│   ├── telco_churn.csv          # Raw dataset (IBM Telco)

│   ├── X_test.csv               # Test features

│   ├── shap_values_xgb.npy      # Precomputed SHAP values

│   ├── divergence_results.csv   # SHAP vs LIME comparison

│   └── fairness_summary.csv     # Fairness audit results

├── models/

│   ├── xgboost_model.pkl        # Trained XGBoost classifier

│   ├── logistic_regression.pkl  # Baseline LR model

│   ├── explainer_xgb.pkl        # SHAP TreeExplainer

│   └── fairness_summary.pkl     # Fairness metrics

├── notebooks/

│   ├── model_training.ipynb     # Data prep + model training

│   ├── shap_analysis.ipynb      # SHAP global + local explanations

│   ├── lime_analysis.ipynb      # LIME + divergence analysis

│   └── fairness_analysis.ipynb  # Fairlearn fairness audit

├── app.py                       # Streamlit dashboard

└── README.md
---

## 📊 Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7939 | 0.5927 | 0.8345 |
| XGBoost | 0.7903 | 0.5731 | 0.8346 |

> Note: Accuracy is intentionally not the primary metric. The goal is
> *explainability* — a model worth explaining, not just a high accuracy number.

---

## 🧠 Explainability Methods

### SHAP (SHapley Additive exPlanations)
- Based on cooperative game theory — Shapley values fairly distribute
  prediction credit across all features
- **Global:** Which features matter most across all customers?
- **Local:** Why was *this specific customer* predicted to churn?
- Uses `TreeExplainer` for XGBoost — exact values, not approximations

### LIME (Local Interpretable Model-agnostic Explanations)
- Fits a local linear model around each individual prediction
- Approximates complex model behaviour in a small neighbourhood
- Model-agnostic — works with any classifier

### SHAP vs LIME Divergence Score
A custom metric developed for this project:
divergence_score = 5 - |overlap between SHAP top-5 and LIME top-5 features|
- Score 0 = perfect agreement
- Score 5 = total disagreement
- High divergence → prediction near decision boundary → lower trust

---

## ⚖️ Fairness Audit

Audited using **Fairlearn** across two sensitive attributes:

| Metric | Gender | Senior Citizen |
|---|---|---|
| Demographic Parity Difference | 0.0301 | 0.2309 |
| Equalized Odds Difference | — | 0.1722 |

**Interpretation:** Senior citizens were predicted to churn at a 14.2% higher rate than non-seniors,
  the largest fairness gap identified. Gender showed negligible disparity (0.032).
  Recommend monitoring senior cohort predictions before production deployment.

---

## 🚀 Run Locally

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/churn-explainability.git
cd churn-explainability

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
# 1. notebooks/model_training.ipynb
# 2. notebooks/shap_analysis.ipynb
# 3. notebooks/lime_analysis.ipynb
# 4. notebooks/fairness_analysis.ipynb

# Launch dashboard
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & Modeling | pandas, scikit-learn, XGBoost |
| Explainability | SHAP, LIME |
| Fairness | Fairlearn |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |

---

## 💡 Key Takeaways

1. **Accuracy alone is not enough** — a model deployed without explanation
   is a liability, not an asset
2. **SHAP and LIME are complementary, not redundant** — their disagreement
   is itself a signal worth surfacing
3. **Fairness is not binary** — demographic parity and equalized odds can
   conflict; the goal is transparency, not perfection
4. **Explainability enables action** — knowing *why* a customer is predicted
   to churn lets retention teams intervene meaningfully

---

## 📚 References

- Lundberg & Lee (2017) — [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- Ribeiro et al. (2016) — ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- Bird et al. (2020) — [Fairlearn: A toolkit for assessing and improving fairness in AI](https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/)
- IBM Telco Customer Churn Dataset — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---
