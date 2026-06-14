## SHAP Findings — Day 2

### Top Features (XGBoost)
- tenure: High tenure → strong negative SHAP (less likely to churn)
- Contract: Month-to-month → pushes churn probability up significantly
- TotalCharges: Higher charges → varies by customer segment

### Top Features (Logistic Regression)
- (fill in after running Cell 9)

### XGBoost vs LR — First Observations
- LR and XGBoost agree on top 3 features: tenure, Contract, MonthlyCharges
- LR assigns more uniform weights; XGBoost shows sharper contrasts
- (add more as you observe)

### Interesting Customers
- Customer [idx]: High tenure but still churning — MonthlyCharges is the culprit
- (fill in after running Cell 7)

---------------------------------------------------------------------

## LIME Findings — Day 3

### SHAP vs LIME Agreement
- Both methods agreed on tenure and Contract as top predictors
- Disagreement observed on mid-tier features (MonthlyCharges vs TotalCharges)

### High Divergence Cases
- Customer [X]: Divergence score 4/5 — SHAP flagged tenure as dominant,
  LIME flagged MonthlyCharges. Customer has low tenure but moderate charges,
  suggesting both features are boundary cases for this prediction.

### What Divergence Means
- High divergence = the prediction sits near a decision boundary
- Neither explanation should be fully trusted in isolation
- Recommendation: flag high-divergence predictions for human review

### Business Implication
- Low divergence predictions → high confidence, can automate retention action
- High divergence predictions → uncertain, needs human judgment

---------------------------------------------------------------------

## Fairness Audit — Day 5

### Demographic Parity (Gender)
- Difference: [fill from Cell 4 output]
- Interpretation: [Low/High] — model predicts churn at [similar/different] 
  rates for male vs female customers

### Demographic Parity (Senior Citizen)
- Difference: [fill from Cell 5 output]
- Senior citizens are predicted to churn at a [higher/lower] rate
- This [may/may not) reflect real behaviour — warrants investigation

### Equalized Odds (Senior Citizen)
- Difference: [fill from Cell 6 output]
- Model makes [more/similar] errors on senior vs non-senior customers

### Key Takeaway
- Fairness and accuracy trade off — a perfectly accurate model
  is not always a fair one
- Recommend: monitor senior citizen predictions before deployment
  and consider reweighting training data if bias is confirmed