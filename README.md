# Smart Student Performance Prediction System

[cite_start]This project, developed at **Zewail City of Science and Technology** [cite: 5, 6] [cite_start]for the **CIE 417: Machine Learning** course [cite: 7][cite_start], implements a multi-tier predictive framework to identify student success and academic risk[cite: 32]. [cite_start]The system transforms multi-dimensional student data into actionable insights for early academic intervention[cite: 30, 471].

---

## 📌 Project Overview

[cite_start]Academic struggles are often identified only after a student has already failed[cite: 29]. [cite_start]This project bridges that gap by predicting student performance mid-term through three primary outputs[cite: 31, 32]:
* [cite_start]**Risk Assessment**: Binary classification to determine if a student will **Pass or Fail**[cite: 33, 43].
* [cite_start]**Progress Measurement**: Regression analysis to estimate the exact **Final Score** (0–100)[cite: 34, 41].
* [cite_start]**Performance Tiers**: Multiclass classification to predict **Final Letter Grades** (A, B, C, D, F)[cite: 35, 42].

---

## 📊 Dataset Specifications

[cite_start]The system utilizes a dataset of **20,000 observations** with **38 independent variables**[cite: 39]. [cite_start]These features are categorized into five primary domains[cite: 45]:

| Category | Features | Count |
| :--- | :--- | :--- |
| **Demographic** | [cite_start]Age, gender, parent income, siblings, commute time[cite: 47]. | 7 |
| **Academic History** | [cite_start]Previous GPA, failed courses, math/science/language background[cite: 47]. | 10 |
| **Behavioral** | [cite_start]Attendance, assignment rate, quiz/midterm scores, library visits[cite: 47]. | 10 |
| **Psychological** | [cite_start]Stress level, sleep hours, motivation, exam anxiety[cite: 47]. | 6 |
| **Institutional** | [cite_start]Course difficulty, teacher experience, class size[cite: 47]. | 5 |



[cite_start]The numerical target **final_score** follows a near-normal distribution[cite: 59]. [cite_start]Categorical targets show significant class imbalance, heavily skewed toward "Pass" and grade "C"[cite: 99, 100].

---

## 🛠️ Methodology & Architecture

### 1. Data Preprocessing
* [cite_start]**Missing Values**: Numerical features are handled using **KNN Imputation**[cite: 129]. [cite_start]Categorical missingness is treated as a distinct "Unknown" state[cite: 131].
* [cite_start]**Outlier Correction**: Logical errors, such as attendance rates exceeding 100% or negative sleep hours, are erased and treated as NaN for imputation[cite: 138, 139, 140].
* [cite_start]**Encoding**: Numerical features undergo **Z-score Standardization**, while categorical features use **One-Hot Encoding**[cite: 143, 144].
* [cite_start]**Class Imbalance**: **SMOTE** (Synthetic Minority Over-sampling Technique) is employed to handle underrepresented classes like "Fail" and rare grades (A/F)[cite: 101, 102].

### 2. Modeling Approaches
The project evaluates two distinct architectural philosophies:

#### **A. Multi-Pipeline Supervised Approach**
[cite_start]Each target is handled by a specialized pipeline[cite: 175]:
* [cite_start]**Regression (Final Score)**: **LassoCV** emerged as the best performer ($R^2 \approx 0.77$), effectively reducing the feature space to 21 significant variables[cite: 236, 294].
* [cite_start]**Multiclass (Final Grade)**: **Gradient Boosting (XGBoost)** significantly outperformed its counterparts with an accuracy of 0.67[cite: 363, 365].
* [cite_start]**Binary (Pass/Fail)**: **Gradient Boosting** achieved the highest overall performance with an **Accuracy of 0.93** and **ROC-AUC of 0.98**[cite: 409, 410].

#### **B. Semi-Supervised Clustering-based Pseudo-Labeling**
[cite_start]To leverage the 600 observations with missing target labels, an inductive clustering algorithm was used[cite: 152, 162]:
1.  [cite_start]**Inductive Clustering**: K-Means centroids are fitted only on the known training set[cite: 165].
2.  [cite_start]**Purity Filtering**: Pseudo-labels are assigned to unlabeled samples only if the cluster purity meets a **90% threshold**[cite: 170].
3.  [cite_start]**Result**: This approach expanded the training corpus while maintaining high label integrity[cite: 171].

---

## 📈 Key Findings
* [cite_start]**Academic Predictors**: Across all models, **Midterm Scores** ($r = 0.6545$) and **Quiz Averages** were the most powerful predictors of success[cite: 111, 460].
* [cite_start]**Socioeconomic Influence**: A strong positive correlation ($r = 0.6822$) exists between **Parent Income** and **Online Portal Usage**[cite: 119].
* [cite_start]**Historical Overrides**: Current-term behavior (attendance and assessment performance) was found to be more predictive than a student's historical GPA or high school grades[cite: 226, 461].

---

## 💻 Tech Stack
* [cite_start]**Language**: Python[cite: 478, 485].
* [cite_start]**Libraries**: `scikit-learn` (Imputers, Linear Models, SVM), `xgboost`, `imblearn` (SMOTE), `pandas`, `numpy`[cite: 479, 485, 489, 737].

---

Would you like me to extract the specific Python code for the **Semi-Supervised Clustering** pipeline or the **XGBoost** implementation?# Student Performance Prediction

