# Customer Churn Prediction – Model Development, Validation, and Deployment
**Course Code:** 21AIC401T  
**Course Name:** Inferential Statistics and Predictive Analytics  
**Institution:** SRM University – Department of Computational Intelligence, School of Computing  
**Assignment Type:** Case Study-Based Modeling Project  
**Total Marks:** 25  
**Submission Deadline:** 10.11.2025  

---

## Project Overview
Customer churn represents a major challenge for telecom and subscription-based industries. This project develops a **predictive machine learning model** to identify customers likely to churn, using the **Telco Customer Churn dataset** from Kaggle.

The project demonstrates the complete lifecycle of a data analytics pipeline — including **data cleaning, EDA, CHAID rule induction, logistic regression modeling, model evaluation, and deployment.**

---

## Objective
To build, validate, compare, and deploy a **Customer Churn Prediction Model** using statistical and predictive modeling concepts such as:
- Data inference and feature correlation
- Model validation and evaluation metrics
- CHAID rule extraction and interpretation
- Logistic regression comparison
- Model deployment and updating for future data

---

## Repository Structure
```
Customer-Churn-Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_model.ipynb
│
├── models/
│   └── best_churn_model.pkl
│
├── visuals/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── roc_auc_comparison.png
│   └── lift_chart.png
│
├── report/
│   └── Customer_Churn_Report.pdf
│
└── README.md
```

---

## Tools & Technologies
| Category | Tool |
|-----------|------|
| Programming | Python (Jupyter Notebook) |
| Libraries Used | pandas, numpy, scikit-learn, seaborn, matplotlib |
| Modeling Algorithms | CHAID (Decision Tree), Logistic Regression |
| Deployment | Pickle |
| Visualization | Matplotlib, Seaborn |
| Dataset Source | [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

---

## Project Workflow

### **1️⃣ Data Preparation & EDA (5 Marks)**
- Dataset loaded and cleaned (missing values, outliers, duplicates removed).  
- Target variable: `Churn` (1 = Yes, 0 = No).  
- Performed Exploratory Data Analysis (EDA) using statistical plots:  
  - Churn distribution  
  - Correlation heatmap  
  - Tenure vs. Churn boxplot  
- Identified key influencing features like **contract type**, **tenure**, and **monthly charges**.

---

### **2️⃣ Model Development using CHAID (5 Marks)**
- Implemented a CHAID-like Decision Tree (`criterion='entropy'` in scikit-learn).  
- Extracted interpretable rules such as:  
  > “If contract type is month-to-month and tenure < 12 months, churn probability is high.”  
- Provided business interpretation of decision rules.

---

### **3️⃣ Model Comparison & Evaluation (5 Marks)**
- Built and evaluated both **CHAID** and **Logistic Regression** models.  
- Compared based on:  
  - Accuracy  
  - Precision, Recall, F1  
  - ROC-AUC Curve  
  - Lift and Gain Chart  

| Model | Accuracy | ROC-AUC | Remarks |
|--------|-----------|----------|----------|
| CHAID | ~0.79 | ~0.84 | Easy interpretability |
| Logistic Regression | ~0.82 | ~0.87 | Better generalization & stability |

---

### **4️⃣ Model Deployment & Updating (5 Marks)**
- Best model (`Logistic Regression`) exported using **Pickle** as `best_churn_model.pkl`.  
- Demonstrated reloading and prediction on new data.  
- Created `update_model()` function to retrain with future customer datasets for model updates.

---

### **5️⃣ Report & GitHub Submission (5 Marks)**
- Structured 15-page project report (`Customer_Churn_Report.pdf`) containing:  
  - Abstract, Introduction, Data Description, Methodology  
  - EDA Results, Model Development, Evaluation  
  - Deployment Framework, Conclusion, and References  
- GitHub repository includes all files: dataset, notebook, visuals, models, and report.

---

## 📊 Key Results
- **Most significant churn drivers:** Contract type, Tenure, MonthlyCharges, OnlineSecurity  
- **Best Performing Model:** Logistic Regression  
- **ROC-AUC:** 0.87  
- **Interpretability:** CHAID model offers human-readable decision rules for managerial use.

---

## 🚀 How to Run This Project

### **Step 1:** Clone the repository
```bash
git clone https://github.com/koushikkb12/ISPA_Customer_Churn_Predict.git
cd ISPA_Customer_Churn_Predict
```

### **Step 2:** Install dependencies
```bash
pip install -r requirements.txt
```

### **Step 3:** Open the notebook
```bash
jupyter notebook notebooks/churn_model.ipynb
```

### **Step 4:** Run all cells sequentially
- Generates visualizations and evaluation metrics  
- Saves deployed model as `best_churn_model.pkl`

### **Step 5:** Load and test the saved model
```python
import pickle, pandas as pd
model = pickle.load(open('models/best_churn_model.pkl', 'rb'))
sample = pd.DataFrame([...])   # example customer data
prediction = model.predict(sample)
```

---

## 🧱 Future Enhancements
- Integrate model API using Flask or FastAPI  
- Automate retraining pipeline with MLOps tools (Airflow/MLflow)  
- Include XGBoost or RandomForest models for ensemble improvements

---

## ✍️ Contributors
**Student Name:** *A S Koushik Babu*  
**Register Number:** *RA2211047010045*  
**Institution:** SRM University – School of Computing

---

## 📚 References
- Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- IBM SPSS CHAID Documentation  
- Scikit-learn Official Docs – Decision Trees & Logistic Regression

---

✅ **End of Project Submission**  
> *"Predictive analytics is not about predicting the future — it’s about empowering better decisions today."*
