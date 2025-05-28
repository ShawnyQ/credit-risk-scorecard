# 🏦 Credit-risk-scorecard

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.1-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A step-by-step notebook that engineers a **proxy “bad-client” label**, performs feature-rich EDA, and builds a leakage-free **XGBoost credit scorecard** with cross-validation and threshold tuning on **438 k** loan-applicant records.

---

## 🎓 Project Overview
1. Raw CSV → missing-value audit  
2. Days-to-years conversion + weighted **heuristic risk score**  
3. EDA (KDEs, lollipops, correlation, Chi-Square)  
4. One-hot encoding → 50 numeric features  
5. Baseline Logistic Regression (AUC 0.98)  
6. Leakage demo + fix → **XGBoost** (AUC 0.9997)  
7. Precision–Recall tuning for business cut-offs  

---

## ❓ Problem Statement
> **Goal:** classify applicants as **high-risk** or **low-risk** when no historical default flag exists, using only profile data (income, dependents, occupation, etc.).

---

## 🔍 Methodology

| Stage        | Detail                                                         |
|--------------|----------------------------------------------------------------|
| **Feature Eng.** | Convert day counts to years; build 5-rule risk score          |
| **Proxy Label**  | `BAD_CLIENT = 1` if `risk_score > 0.20`                       |
| **EDA**         | KDE + lollipop plots to visualise class separation            |
| **Stats**       | Pearson heat-map, Chi-Square on categoricals                  |
| **Pre-proc**    | Drop ID, impute, binary map, one-hot (50 features)            |
| **Baseline**    | Class-weighted Logistic Regression                            |
| **Tree Model**  | XGBoost + `scale_pos_weight`, 5-fold CV                       |
| **Threshold**   | PR curve to pick 0.30 vs 0.70 cut-offs                        |

---

## 📊 Key Results

| Metric            | Logistic Reg. | XGBoost (Refined) |
|-------------------|---------------|-------------------|
| **ROC AUC**       | 0.9772        | **0.9997**        |
| **Precision (Bad)** | 0.1085      | **0.9738**        |
| **Recall (Bad)**    | 0.9456      | **1.0000**        |
| **F1 (Bad)**        | 0.1947      | **0.9867**        |

*Why almost perfect?* The label is deterministic; a tree can rediscover the rules exactly.  
Swap in real repayment data and retrain for realistic scores.

---

## ⚖️ Business Cut-offs

| Threshold | Precision | Recall | When to Use |
|-----------|-----------|--------|-------------|
| **0.30**  | 0.94      | 1.00   | **Risk-averse** – catch every defaulter, accept more manual reviews |
| **0.70**  | 0.99      | 0.997  | **Cost-focused** – minimise false positives, tolerate a ~0.3 % miss-rate |

---

## Data Access

PS: Please access data in this link here: 
link: https://drive.google.com/file/d/1WeIIugqArR0fdVt7Wpcyj_3KmD4h8Jzb/view?usp=sharing

## 👨‍💻 Author

**Shawn Waringu**  
Data Scientist & Analyst

[LinkedIn](https://www.linkedin.com/in/shawn-chege-856048312)
[GitHub](https://github.com/ShawnyQ)
