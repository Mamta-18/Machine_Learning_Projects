# 💳 Online Payment Fraud Detection using Machine Learning

<p align="center">
  <img src="banner.jpg" width="800"/>
</p>

## 📌 Project Overview

This project focuses on detecting fraudulent online payment transactions using Exploratory Data Analysis (EDA) and Machine Learning techniques. The dataset contains over 1 million transactions, making it a large-scale real-world problem.

The objective is to identify suspicious transactions and build models that can accurately classify fraud and non-fraud cases.

---

## 📊 Dataset Description

* Total Records: **1M+ transactions**
* Features: **11 columns**
* Target Variable: **isFraud (0 = Non-Fraud, 1 = Fraud)**

### Key Features:

* `type` – Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
* `amount` – Transaction amount
* `oldbalanceOrg`, `newbalanceOrig`
* `oldbalanceDest`, `newbalanceDest`

---

## 🧹 Data Preprocessing

* Removed duplicate records
* Handled categorical variables using one-hot encoding
* Dropped irrelevant features (`nameOrig`, `nameDest`)
* Created new features:

  * `balance_diff_orig`
  * `balance_diff_dest`
* Selected only numerical features for modeling

---

## 📊 Exploratory Data Analysis (EDA)

* Identified class imbalance (fraud cases < 1%)
* Fraud mostly occurs in:

  * **TRANSFER**
  * **CASH_OUT**
* High-value transactions are more likely to be fraudulent
* Strong relationships observed between balance-related features

---

## ⚖️ Handling Imbalanced Data

* Applied **undersampling technique**
* Balanced dataset for better model performance

---

## 🤖 Machine Learning Models

### 🔹 Logistic Regression

* Accuracy: **~92%**
* Fraud Recall: **97%**
* Successfully detected most fraudulent transactions

### 🔹 Isolation Forest (Anomaly Detection)

* Used for detecting outliers
* Performance was lower due to overlapping patterns between fraud and non-fraud transactions

---

## 📈 Model Evaluation

* Used:

  * Accuracy
  * Precision
  * Recall
  * F1-Score

### Key Insight:

> Recall is more important than accuracy in fraud detection because missing a fraud transaction can lead to financial loss.

---

## 📊 Results & Insights

* Logistic Regression outperformed Isolation Forest
* Feature engineering significantly improved model performance
* Fraud transactions are not always extreme outliers

---

## 📊 Dashboard

An interactive Power BI dashboard was created to visualize:

* Fraud vs Non-Fraud distribution
* Fraud by transaction type
* Transaction amount patterns
* High-risk transaction insights

---

## 🛠️ Tools & Technologies

* Python (Pandas, NumPy)
* Data Visualization (Matplotlib, Seaborn)
* Machine Learning (Scikit-learn)
* Power BI

---

## 🚀 Future Improvements

* Use SMOTE for better imbalance handling
* Try advanced models (Random Forest, XGBoost)
* Deploy model using Streamlit or Flask
* Real-time fraud detection system

---

## 📌 Conclusion

This project demonstrates how machine learning and data analysis can be used to detect fraud effectively. It highlights the importance of handling imbalanced data and selecting the right model for the problem.

---

## 👩‍💻 Author

**Mamta Rathore**
Aspiring Data Analyst | Python | SQL | Power BI
