# Risk Level Prediction using Machine Learning

## 📌 Overview
This project presents a **complete end-to-end machine learning pipeline** for **multi-class risk level prediction** using structured (tabular) data.  
It demonstrates real-world ML practices including **data cleaning, class imbalance handling, feature correlation analysis**, and **comparative evaluation of multiple machine learning models**.

The project is designed for **learning, experimentation, and portfolio demonstration**.

## 🎯 Problem Statement
Given a dataset containing multiple features, the objective is to **predict the risk level** of each instance into one of the following categories:
- **Low Risk (0)**
- **Mid Risk (1)**
- **High Risk (2)**
  
## 🛠️ Technologies & Libraries
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- XGBoost  
- Matplotlib  
- Seaborn  

## 📂 Project Structure
Risk-Level-Prediction/
│
├── Preprocessing.py
├── Correlation.py
├── Logistic_regression.py
├── Decisiontree.py
├── Randomforest.py
├── Supportvectormachine.py
├── Xg_Boost.py
├── data.csv
├── results/
│ └── *.png
└── README.md

## 🔄 Project Workflow

### 1️⃣ Data Preprocessing
Implemented in **`Preprocessing.py`**:

- Load dataset from `data.csv`
- Remove duplicate rows
- Inspect missing values
- Remove records belonging to the age group **10–18**
- Encode target variable:
  - `low risk → 0`
  - `mid risk → 1`
  - `high risk → 2`
- Split data into training and testing sets (80:20)
- Handle class imbalance using **SMOTE**

This step ensures the dataset is **clean, balanced, and model-ready**.

### 2️⃣ Correlation Analysis & Feature Reduction
Implemented in **`Correlation.py`**:

- Compute feature correlation matrix
- Visualize correlations using a heatmap
- Identify highly correlated feature pairs (|correlation| > 0.4)
- Remove redundant features to reduce multicollinearity
- Train Logistic Regression on reduced features
- Evaluate using accuracy, classification report, and confusion matrix

## 🤖 Machine Learning Models

### 3️⃣ Logistic Regression
Implemented in **`Logistic_regression.py`**:

- Multi-class classification
- Trained on SMOTE-balanced data
- Evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix
- Multi-class ROC curves with AUC scores

### 4️⃣ Decision Tree Classifier
Implemented in **`Decisiontree.py`**:

- Trained on balanced training data
- Captures non-linear feature interactions
- Evaluation using:
  - Accuracy
  - Classification report
  - Confusion matrix
- Multi-class ROC-AUC analysis

### 5️⃣ Random Forest Classifier
Implemented in **`Randomforest.py`**:

- Ensemble-based learning approach
- Improves generalization by combining multiple trees
- Evaluated using:
  - Accuracy
  - Confusion matrix
  - Classification report
  - ROC curves

### 6️⃣ Support Vector Machine (SVM)
Implemented in **`Supportvectormachine.py`**:

- Margin-based classification
- Effective in high-dimensional feature spaces
- Evaluated using:
  - Accuracy
  - Confusion matrix
  - Classification report
  - ROC curves (multi-class)

### 7️⃣ XGBoost Classifier
Implemented in **`Xg_Boost.py`**:

- Gradient boosting-based ensemble model
- Handles complex non-linear relationships
- Evaluated using:
  - Accuracy
  - Classification report
  - Confusion matrix
  - Multi-class ROC curves with AUC scores

## 📊 Evaluation Metrics
All models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve and AUC Score (multi-class)

This enables **fair and consistent comparison** across models.

## 📈 Results
- SMOTE effectively resolves class imbalance
- Correlation-based feature selection reduces redundancy
- Logistic Regression provides an interpretable baseline
- Tree-based models capture non-linear patterns
- XGBoost delivers strong performance on complex relationships

All plots and visual outputs are stored in the `results/` directory.


## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Risk-Level-Prediction.git
cd Risk-Level-Prediction
```

2. Install Dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

3. Add Dataset

Place data.csv in the project root directory.

4. Run the Scripts (Recommended Order)
python Preprocessing.py
python Correlation.py
python Logistic_regression.py
python Decisiontree.py
python Randomforest.py
python Supportvectormachine.py
python Xg_Boost.py

🎯 Applications

Risk assessment systems
Healthcare decision support
Insurance and financial risk analysis
Machine learning benchmarking
Academic and resume portfolio projects

🚀 Future Enhancements

Hyperparameter tuning
Cross-validation
Feature importance visualization
Model comparison table
Pipeline automation

👤 Author
Harsh Bansal
