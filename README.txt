🧠 HR Analytics — Employee Attrition Prediction
==================================================

A complete Machine Learning classification project predicting employee attrition using the HR Employee Attrition dataset.
The notebook demonstrates the full ML workflow in Python / scikit-learn, from data exploration to model selection and evaluation.

--------------------------------------------------
⚙️ PROJECT STRUCTURE
--------------------------------------------------

1) DATA UNDERSTANDING & EDA
- Import, clean, and segment employee data
- Analyze personal, work, reward, and satisfaction variables
- Visualize distributions and correlations

2) PREPROCESSING PIPELINE
- Missing values: SimpleImputer
- Scaling: StandardScaler (numeric)
- Encoding: OneHotEncoder (categorical)
- Combined using ColumnTransformer
- Applied consistently through Pipeline

3) MODEL EVALUATION WORKFLOW
- Basic Test → Train/test split
- Cross-Validation → 5-fold stability check
- Grid Search → Hyperparameter tuning
- Feature Selection (RFE) → tested only on Logistic Regression (no improvement)

4) MODELS TESTED
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

--------------------------------------------------
📊 RESULTS SUMMARY
--------------------------------------------------

| Model | Best F1 (CV) | F1 (Test) | Accuracy | Recall | Precision |
|--------|--------------|-----------|-----------|----------|-------------|
| Logistic Regression | 0.49 | 0.53 | 0.89 | 0.46 | 0.62 |
| KNN | 0.29 | 0.17 | 0.87 | 0.10 | 0.57 |
| Random Forest | 0.53 | 0.63 | 0.89 | 0.67 | 0.59 |

Best model: RandomForestClassifier
class_weight='balanced', max_depth=5, n_estimators=100, max_features='sqrt'

--------------------------------------------------
🧩 KEY INSIGHTS
--------------------------------------------------
- Logistic Regression and Random Forest performed best.
- KNN underperformed on imbalanced data.
- Random Forest achieved higher recall (better at detecting attrition).
- Feature selection (RFE) did not improve performance.

--------------------------------------------------
🛠️ TOOLS & LIBRARIES
--------------------------------------------------
Python, pandas, numpy, matplotlib, seaborn
scikit-learn, Pipeline, GridSearchCV, RFE

--------------------------------------------------
📁 DATASET
--------------------------------------------------
Source: IBM HR Analytics Employee Attrition Dataset
Target variable: Attrition (Yes = 1 / No = 0)
Records: ~1,470 employees

--------------------------------------------------
🧾 AUTHOR
--------------------------------------------------
Project created as part of a Machine Learning training series (Scikit-Learn Workflow).
All code, visualizations, and evaluations built step-by-step in JupyterLab.
