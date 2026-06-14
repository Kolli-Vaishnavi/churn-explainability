import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ChurnXplain",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #1e1e2e; border-radius: 8px; padding: 12px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }
    .stAlert { border-radius: 8px; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 2rem !important; }
    h3 { color: #a6adc8; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models & Data ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb_model     = joblib.load('models/xgboost_model.pkl')
    lr_model      = joblib.load('models/logistic_regression.pkl')
    scaler        = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer_xgb = joblib.load('models/explainer_xgb.pkl')
    return xgb_model, lr_model, scaler, feature_names, explainer_xgb

@st.cache_data
def load_data():
    X_test          = pd.read_csv('data/X_test.csv').reset_index(drop=True)
    y_test          = pd.read_csv('data/y_test.csv').squeeze().reset_index(drop=True)
    X_train         = pd.read_csv('data/X_train.csv').reset_index(drop=True)
    shap_values_xgb = np.load('data/shap_values_xgb.npy')
    return X_test, y_test, X_train, shap_values_xgb

@st.cache_resource
def get_lime_explainer(_X_train, _feature_names):
    return lime.lime_tabular.LimeTabularExplainer(
        training_data         = _X_train.values,
        feature_names         = _feature_names,
        class_names           = ['No Churn', 'Churn'],
        mode                  = 'classification',
        discretize_continuous = True,
        random_state          = 42
    )

xgb_model, lr_model, scaler, feature_names, explainer_xgb = load_models()
X_test, y_test, X_train, shap_values_xgb                  = load_data()
lime_explainer = get_lime_explainer(X_train, feature_names)

# ─── Helper Functions ──────────────────────────────────────────────────────────
def get_prediction(customer_data):
    prob      = xgb_model.predict_proba(customer_data)[0][1]
    predicted = int(prob >= 0.5)
    return predicted, prob

def get_shap_waterfall(idx):
    shap_exp = shap.Explanation(
        values        = shap_values_xgb[idx],
        base_values   = explainer_xgb.expected_value,
        data          = X_test.iloc[idx].values,
        feature_names = feature_names
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_exp, show=False)
    plt.tight_layout()
    return fig

def get_lime_plot(idx):
    exp        = lime_explainer.explain_instance(
        data_row   = X_test.iloc[idx].values,
        predict_fn = xgb_model.predict_proba,
        num_features = 10,
        num_samples  = 500
    )
    lime_items = exp.as_list()
    features   = [x[0][:35] for x in lime_items]
    weights    = [x[1] for x in lime_items]
    colors     = ['tomato' if w > 0 else 'steelblue' for w in weights]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, weights, color=colors)
    ax.axvline(x=0, color='white', linewidth=0.8)
    ax.set_xlabel('Feature Weight')
    ax.set_title('LIME Local Explanation')
    plt.tight_layout()
    return fig, exp

def get_divergence(idx, lime_exp):
    shap_top5 = pd.Series(
        np.abs(shap_values_xgb[idx]),
        index=feature_names
    ).sort_values(ascending=False).head(5).index.tolist()

    lime_top5 = [f[0] for f in sorted(
        lime_exp.as_list(), key=lambda x: abs(x[1]), reverse=True
    )[:5]]

    overlap = len(set(shap_top5) & set(lime_top5))
    return 5 - overlap, shap_top5, lime_top5

def get_global_shap_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values_xgb,
        X_test,
        feature_names = feature_names,
        plot_type     = "bar",
        show          = False
    )
    plt.tight_layout()
    return fig

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 ChurnXplain")
    st.markdown("**Explainable AI Dashboard**")
    st.markdown("Customer Churn Prediction with SHAP & LIME")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard explains *why* an ML model predicts
    customer churn — not just whether it does.

    **Methods used:**
    - 🔵 SHAP — global + local explanations
    - 🟠 LIME — local linear approximations
    - ⚖️ Divergence analysis
    - 🧑‍⚖️ Fairness audit via Fairlearn
    """)
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(f"**{len(X_test)}** test customers")
    st.markdown(f"**{len(feature_names)}** features")
    st.markdown("Source: Telco Customer Churn (IBM)")
    st.markdown("---")
    st.caption("Built for Google Student Researcher Application")

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 ChurnXplain — Explainable AI for Customer Churn")
st.caption(
    "This dashboard goes beyond accuracy — it explains *why* predictions happen, "
    "compares two explainability methods, and audits for demographic fairness."
)
st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧍 Customer Prediction",
    "🌍 Global Insights",
    "⚖️ SHAP vs LIME",
    "🧑‍⚖️ Fairness Audit"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Customer Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Individual Customer Explanation")
    st.caption("Select a customer to see their churn prediction and why the model made it.")

    col1, col2 = st.columns([1, 3])

    with col1:
        customer_idx = st.slider(
            "Customer Index", 0, len(X_test) - 1, 0
        )

        actual           = y_test.iloc[customer_idx]
        predicted, prob  = get_prediction(X_test.iloc[[customer_idx]])

        st.markdown("---")

        # Confidence gauge
        st.markdown("**Churn Probability**")
        st.progress(float(prob))

        # Color coded prediction
        if predicted == 1:
            st.error(f"🔴 CHURN — {prob:.1%} confidence")
        else:
            st.success(f"🟢 NO CHURN — {1-prob:.1%} confidence")

        st.markdown(f"**Actual:** {'🔴 Churn' if actual == 1 else '🟢 No Churn'}")

        if predicted == actual:
            st.caption("✅ Correct prediction")
        else:
            st.caption("❌ Incorrect prediction")

        st.markdown("---")
        st.markdown("**Customer Features**")
        customer_df          = X_test.iloc[[customer_idx]].T.reset_index()
        customer_df.columns  = ['Feature', 'Value']
        st.dataframe(customer_df, hide_index=True, height=320)

    with col2:
        st.markdown("### SHAP Explanation")
        st.caption(
            "Features in **red** push the prediction toward Churn. "
            "Features in **blue** push toward No Churn. "
            "Bar length = strength of influence."
        )
        with st.spinner("Generating SHAP explanation..."):
            shap_fig = get_shap_waterfall(customer_idx)
            st.pyplot(shap_fig)
            plt.close()

        st.markdown("---")

        st.markdown("### LIME Explanation")
        st.caption(
            "LIME fits a local linear model around this specific customer. "
            "Positive weights push toward Churn. Negative weights push away."
        )
        with st.spinner("Generating LIME explanation..."):
            lime_fig, _ = get_lime_plot(customer_idx)
            st.pyplot(lime_fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Global Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Global Feature Importance")
    st.caption(
        "Average absolute SHAP values across all test customers — "
        "which features drive churn predictions most overall."
    )

    with st.spinner("Generating global SHAP plot..."):
        global_fig = get_global_shap_plot()
        st.pyplot(global_fig)
        plt.close()

    st.markdown("---")
    st.subheader("Dataset Summary")

    preds       = xgb_model.predict(X_test)
    churn_count = int(preds.sum())
    total       = len(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",    total)
    col2.metric("Predicted Churners", churn_count)
    col3.metric("Predicted Retained", total - churn_count)
    col4.metric("Churn Rate",         f"{preds.mean():.1%}")

    st.markdown("---")
    st.markdown("### How to Read SHAP Values")
    st.info("""
    - **High SHAP value** → feature strongly pushed this prediction toward Churn
    - **Low (negative) SHAP value** → feature pushed prediction away from Churn
    - **Near zero** → feature had little effect on this prediction
    - The baseline (expected value) is the average prediction across all customers
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP vs LIME Divergence
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("SHAP vs LIME — Divergence Analysis")
    st.caption(
        "When SHAP and LIME disagree on which features matter most, "
        "the prediction is near a decision boundary and less trustworthy."
    )

    compare_idx = st.slider(
        "Select Customer", 0, len(X_test) - 1, 0, key="compare_slider"
    )

    with st.spinner("Computing divergence..."):
        _, lime_exp_compare       = get_lime_plot(compare_idx)
        div_score, shap_top5, lime_top5 = get_divergence(compare_idx, lime_exp_compare)

    # Divergence badge
    if div_score >= 3:
        st.error(
            f"⚠️ High Divergence: {div_score}/5 — "
            "SHAP and LIME strongly disagree. Treat with caution."
        )
    elif div_score >= 1:
        st.warning(
            f"🟡 Moderate Divergence: {div_score}/5 — "
            "Some disagreement between methods."
        )
    else:
        st.success(
            f"✅ Low Divergence: {div_score}/5 — "
            "SHAP and LIME are in strong agreement."
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SHAP Top 5 Features**")
        for i, f in enumerate(shap_top5):
            icon = "✅" if f in lime_top5 else "❌"
            st.write(f"{i+1}. {icon} `{f}`")

    with col2:
        st.markdown("**LIME Top 5 Features**")
        for i, f in enumerate(lime_top5):
            icon = "✅" if f in shap_top5 else "❌"
            st.write(f"{i+1}. {icon} `{f}`")

    st.markdown("---")
    st.markdown("### Side-by-Side Comparison")

    c1, c2 = st.columns(2)
    with c1:
        shap_fig2 = get_shap_waterfall(compare_idx)
        st.pyplot(shap_fig2)
        plt.close()
    with c2:
        lime_fig2, _ = get_lime_plot(compare_idx)
        st.pyplot(lime_fig2)
        plt.close()

    st.markdown("---")
    st.markdown("### Why Divergence Matters")
    st.info("""
    **SHAP** uses Shapley values from cooperative game theory —
    it distributes prediction credit fairly across all features globally.

    **LIME** fits a local linear model around each prediction —
    it approximates the model behaviour in a small neighbourhood.

    **When they disagree:**
    - The prediction sits near a decision boundary
    - Small changes in feature values could flip the prediction
    - Neither explanation alone should be fully trusted

    **Business rule:**
    - ✅ Divergence 0–1 → automate retention action
    - 🟡 Divergence 2 → flag for review
    - ⚠️ Divergence 3–5 → escalate to human judgment
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Fairness Audit")
    st.caption(
        "Does the model predict churn fairly across demographic groups? "
        "Checking gender and senior citizen status."
    )

    try:
        fairness_summary = joblib.load('models/fairness_summary.pkl')

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Demographic Parity — Gender",
            f"{fairness_summary['demographic_parity_gender']:.3f}",
            help="0 = perfectly fair. >0.1 = concerning disparity."
        )
        col2.metric(
            "Demographic Parity — Senior",
            f"{fairness_summary['demographic_parity_senior']:.3f}",
            help="Difference in churn prediction rates between senior and non-senior."
        )
        col3.metric(
            "Equalized Odds — Senior",
            f"{fairness_summary['equalized_odds_senior']:.3f}",
            help="Difference in error rates across senior groups."
        )

        st.markdown("---")
        st.markdown("### Verdict")

        dpd_g = fairness_summary['demographic_parity_gender']
        dpd_s = fairness_summary['demographic_parity_senior']

        if abs(dpd_g) < 0.05 and abs(dpd_s) < 0.1:
            st.success(
                "✅ Model shows **low bias** across gender and senior status. "
                "Safe to deploy with continued monitoring."
            )
        elif abs(dpd_s) >= 0.1:
            st.error(
                f"⚠️ Senior citizens are predicted to churn at a significantly "
                f"different rate (difference: {dpd_s:.3f}). "
                f"This may reflect real behaviour or model bias — investigate before deploying."
            )
        else:
            st.warning(
                "🟡 Minor disparities detected. Monitor post-deployment."
            )

        st.markdown("---")
        st.markdown("### Fairness Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.image('data/fairness_churn_rates.png',
                     caption="Predicted churn rate by demographic group")
        with col2:
            st.image('data/fairness_by_senior.png',
                     caption="Accuracy, Precision, Recall by Senior Status")

        st.markdown("---")
        st.markdown("### Understanding Fairness Metrics")
        st.info("""
        **Demographic Parity Difference**
        Are churn predictions made at equal rates across groups?
        Value near 0 = fair. Above 0.1 = investigate.

        **Equalized Odds Difference**
        Are prediction errors equally distributed across groups?
        High values = model systematically fails one group more.

        **The fairness-accuracy tradeoff:**
        No model can simultaneously satisfy all fairness definitions.
        The goal here is transparency and informed deployment — not perfection.
        """)

    except FileNotFoundError:
        st.error(
            "Fairness data not found. "
            "Please run notebooks/fairness_analysis.ipynb first."
        )