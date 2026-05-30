# Credit Card Fraud Pattern Analysis

   > Identifying behavioural fraud signals in 284,807 credit card transactions using machine learning — addressing extreme class imbalance (99.83% legitimate) to achieve 94% fraud recall.

Machine learning analysis of 284,807 credit card transactions to detect fraud patterns, addressing severe class imbalance using SMOTE and logistic regression.

## Results
- **Fraud recall: 94%** — model correctly identified 94% of actual fraud cases
- **Class imbalance**: 99.83% legitimate vs 0.17% fraud, handled via SMOTE resampling
- **Primary risk signals**: transaction timing and amount identified as top fraud predictors via feature importance analysis

## Why This Matters

   In production fraud detection systems, **recall on fraud cases is the metric that matters most** — a missed fraud case is a direct financial loss, while a false positive is merely an inconvenience. This analysis demonstrates how SMOTE resampling combined with careful feature engineering can achieve high recall even under extreme class imbalance, and surfaces the specific transaction features (timing patterns, amount thresholds) that drive fraud risk.

## Charts
| Class Imbalance | Fraud Rate by Hour |
|---|---|
| ![](charts/class_imbalance.png) | ![](charts/fraud_by_hour.png) |

| Confusion Matrix | Feature Importance |
|---|---|
| ![](charts/confusion_matrix.png) | ![](charts/feature_importance.png) |

## Approach
1. **Exploratory analysis** — visualised class imbalance and transaction patterns by hour and amount
2. **Feature engineering** — scaled `Amount` and `Time` using StandardScaler; extracted hour-of-day as a risk signal
3. **SMOTE resampling** — synthetically oversampled fraud cases to balance training data
4. **Logistic regression** — trained baseline classifier optimised for recall (minimising missed fraud)
5. **Feature importance** — analysed model coefficients to surface primary behavioural risk signals

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (LogisticRegression, StandardScaler, train_test_split)
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## Dataset
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group via Kaggle.
Download `creditcard.csv` and place in the root directory before running.

## Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python fraud_script.py
```
