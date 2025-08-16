# 🛂 EasyVisa — Predicting Visa Case Outcomes with Machine Learning
*An advanced ML pipeline to automate visa application screening, optimize approvals, and generate data-driven insights for labor certification.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-EDA-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-yellow.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Viz-teal.svg)](https://seaborn.pydata.org/)
[![Imbalanced-learn](https://img.shields.io/badge/ImbalancedLearn-SMOTE-red.svg)](https://imbalanced-learn.org/)

---

## 📌 Problem Statement & Context

The U.S. visa application process is highly competitive, with significant socio-economic consequences for applicants and employers. Predicting the likelihood of visa case approval or denial can help streamline decision-making, reduce uncertainties, and guide applicants and organizations in preparing stronger submissions.

Manual reviews are **time-intensive**, creating bottlenecks in processing. To improve efficiency, OFLC partnered with **EasyVisa** to leverage **machine learning** to:
- Predict whether a visa application will be **Certified** or **Denied**.
- Identify key **predictors of approvals**.
- Support policymakers and employers with **data-driven decisions**.

The ultimate goal is to build models that not only achieve strong accuracy but also maintain balanced recall and precision, especially for the minority class (“Denied”), where misclassification could have serious implications.

---

## 🎯 Objective
Develop a **classification model** to predict visa case outcomes (`Certified` vs. `Denied`) and uncover insights into applicant and employer factors influencing decisions.

### Dataset Features
- **case_id**: Unique application ID  
- **continent**: Applicant’s continent of origin  
- **education_of_employee**: Education level (High School, Bachelor’s, Master’s, Doctorate)  
- **has_job_experience**: Prior work experience (Y/N)  
- **requires_job_training**: Whether job requires training (Y/N)  
- **no_of_employees**: Employer size  
- **yr_of_estab**: Year company established  
- **region_of_employment**: US region of employment  
- **prevailing_wage**: Benchmark wage for role  
- **unit_of_wage**: Hourly, Weekly, Monthly, Yearly  
- **full_time_position**: Full-time (Y/N)  
- **case_status**: Target variable (Certified/Denied)  

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings
- **Education:** Approval probability rises with education — from **63% (Bachelor’s)** to **86% (Doctorate)**.  
- **Experience:** Experienced applicants had ~**75% approval**, compared to ~**50%** for non-experienced.  
- **Geography:**  
  - **Asia:** ~66% approval  
  - **Europe:** ~81% approval  
  - **US Midwest region:** ~75% approval (highest among regions).  
- **Training:** Requirement for training showed **minimal influence** on approval.  
- **Full-time vs. Part-time:** Both saw approval rates above **66%**.  
- **Company Metrics (size, year established, wages):** Had limited predictive power.  

---

## 🤖 Modeling Approach
- **Models Tested:**
  -- Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Bagging, XGBoost  
- **Imbalance Handling:** SMOTE, Random Under-Sampling, Tomek Links  
- **Evaluation Metrics:** Accuracy, Recall, Precision, F1-score, ROC-AUC  
- **Hyperparameter Tuning:** GridSearchCV & RandomizedSearchCV for boosting models  

---

## 📊 Model Results (Illustrative)

| Model            | Accuracy | Recall | Precision | F1 Score | Notes |
|------------------|----------|--------|-----------|----------|-------|
| Decision Tree    | ~78%     | ~75%   | ~77%      | ~76%     | High interpretability |
| Random Forest    | ~83%     | ~80%   | ~82%      | ~81%     | Robust, less variance |
| Gradient Boosting| ~85%     | ~82%   | ~84%      | ~83%     | Balanced trade-offs |
| **XGBoost** ✅   | **87%**  | **85%**| **86%**   | **85%**  | Best overall performer |

*(Insert your exact numbers here once finalized from the notebook.)*

---

## 💡 Business Insights & Recommendations
1. **Prioritize Highly Educated Applicants**  
   - Doctorate and Master’s holders show the **highest approval rates**.  
   - Suggest fast-tracking highly educated workers to fill **specialized roles**.  

2. **Leverage Experience as a Key Driver**  
   - Work experience boosts approval by ~25 percentage points.  
   - Employers should emphasize relevant prior work when applying.  

3. **Regional Considerations**  
   - Midwest and South US regions have higher approval rates.  
   - Policymakers can encourage applications targeting these geographies.  

4. **Wage & Training Neutrality**  
   - Prevailing wages and training requirements had **little impact** on outcomes.  
   - Suggest policy efforts focus more on **skills & education** than wage adjustments.  

5. **Balanced Campaign Targeting**  
   - By using ML model scores, EasyVisa can rank applicants by **approval likelihood**.  
   - High-score applicants: **priority for manual review**.  
   - Low-score applicants: **automated processing** to save time.  
