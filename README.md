# Smart Student Performance Prediction System

This project, developed at **Zewail City of Science and Technology** for the **CIE 417: Machine Learning** course, implements a multi-tier predictive framework to identify student success and academic risk. The system transforms multi-dimensional student data into actionable insights for early academic intervention.

---

## Collaborators


- [Aml Ismail](https://github.com/Aml-Ismail)
- [Youssef Allam](https://github.com/YoussefMAllam)
- [Mohammad AbdelRahman](https://github.com/MomoAbdelRahman)


## 📌 Project Overview

Academic struggles are often identified only after a student has already failed. This project bridges that gap by predicting student performance mid-term through three primary outputs:
* **Risk Assessment**: Binary classification to determine if a student will **Pass or Fail**.
* **Progress Measurement**: Regression analysis to estimate the exact **Final Score** (0–100).
* **Performance Tiers**: Multiclass classification to predict **Final Letter Grades** (A, B, C, D, F).

---

## 📊 Dataset Specifications

The system utilizes a dataset of **20,000 observations** with **38 independent variables**. These features provide a multi-dimensional view of the student experience:

| Category | Features | Count |
| :--- | :--- | :--- |
| **Demographic** | Age, gender, parent income, siblings, commute time. | 7 |
| **Academic History** | Previous GPA, failed courses, math/science/language background. | 10 |
| **Behavioral** | Attendance, assignment rate, quiz/midterm scores, library visits. | 10 |
| **Psychological** | Stress level, sleep hours, motivation, exam anxiety. | 6 |
| **Institutional** | Course difficulty, teacher experience, class size. | 5 |



The numerical target **final_score** follows a near-normal distribution. Categorical targets show significant class imbalance, heavily skewed toward "Pass" and grade "C".

---

## 🛠️ Methodology & Architecture

### 1. Data Preprocessing
* **Missing Values**: Numerical features are handled using **KNN Imputation**. Categorical missingness is treated as a distinct "Unknown" state.
* **Outlier Correction**: Logical errors, such as attendance rates exceeding 100% or negative sleep hours, are erased and treated as NaN for imputation.
* **Encoding**: Numerical features undergo **Z-score Standardization**, while categorical features use **One-Hot Encoding**.
* **Class Imbalance**: **SMOTE** (Synthetic Minority Over-sampling Technique) is employed to handle underrepresented classes like "Fail" and rare grades (A/F).

### 2. Modeling Approaches
The project evaluates two distinct architectural philosophies:

#### **A. Multi-Pipeline Supervised Approach**
Each target is handled by a specialized pipeline:
* **Regression (Final Score)**: **LassoCV** emerged as the best performer ($R^2 \approx 0.77$), effectively reducing the feature space to 21 significant variables.
* **Multiclass (Final Grade)**: **Gradient Boosting (XGBoost)** significantly outperformed its counterparts with an accuracy of 0.67.
* **Binary (Pass/Fail)**: **Gradient Boosting** achieved the highest overall performance with an **Accuracy of 0.93** and **ROC-AUC of 0.98**.

#### **B. Semi-Supervised Clustering-based Pseudo-Labeling**
To leverage the 600 observations with missing target labels, an inductive clustering algorithm was used:
1.  **Inductive Clustering**: K-Means centroids are fitted only on the known training set.
2.  **Purity Filtering**: Pseudo-labels are assigned to unlabeled samples only if the cluster purity meets a **90% to 95% threshold**.
3.  **Result**: This approach expanded the training corpus while maintaining high label integrity.

---

## 📈 Key Findings
* **Academic Predictors**: Across all models, **Midterm Scores** ($r = 0.6545$) and **Quiz Averages** were the most powerful predictors of success.
* **Socioeconomic Influence**: A strong positive correlation exists between **Parent Income** and **Online Portal Usage** ($r = 0.6822$).
* **Historical Overrides**: Current-term behavior (attendance and assessment performance) was found to be more predictive than a student's historical GPA or high school grades.

---

## 💻 Tech Stack
* **Language**: Python.
* **Libraries**: `scikit-learn` (Imputers, Linear Models, SVM), `xgboost`, `imblearn` (SMOTE), `pandas`, `numpy`.

---
