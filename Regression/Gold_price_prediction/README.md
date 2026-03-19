


# 🟡 Gold Price Prediction using Machine Learning

This project aims to predict **Gold ETF (GLD)** prices using machine learning techniques based on financial market indicators. It includes **data preprocessing, exploratory data analysis, model building, evaluation**, and a **prediction system** using a **Random Forest Regressor**.

---

## 📁 Project Structure

```

Gold Price Prediction
│── gld_price_data.csv
│── gold_prediction.ipynb / gold_prediction.py
│── README.md

````

---

## 🚀 Objective

To develop a machine learning model that can accurately predict **GLD (Gold Price)** based on the following indicators:

- **SPX** – S&P 500 Index  
- **USO** – Crude Oil Price  
- **SLV** – Silver Price  
- **EUR/USD** – Currency Exchange Rate  

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
````

---

## 📊 Dataset Overview

* Rows: **2290**
* Columns:

  * Date
  * SPX
  * GLD
  * USO
  * SLV
  * EUR/USD

### ✔ Data Preprocessing Steps

* Converted **Date** to datetime format
* Verified **null values** (none found)
* Displayed structure using `.info()` and `.describe()`

---

## 📈 Correlation Analysis

A heatmap was used to analyze the relationship between variables.

### 🔍 Key Insights

* **GLD & SLV** → Strong positive correlation (**0.86**)
* **GLD & USO** → Weak negative correlation
* **GLD & SPX** → Very low correlation

---

## 📉 Data Distribution

Distribution of GLD price:

```python
sns.distplot(Gold_Data['GLD'], color='green')
```

---

## ✂️ Feature Selection

### **Features (X)**

* SPX
* USO
* SLV
* EUR/USD

### **Target (y)**

* GLD

Data split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
```

---

## 🤖 Model Building – Random Forest Regressor

```python
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 📈 Model Evaluation

| Metric       | Score      |
| ------------ | ---------- |
| **MAE**      | 1.27       |
| **MSE**      | 6.51       |
| **RMSE**     | 2.55       |
| **R² Score** | **0.9876** |

### ✔ Interpretation

* Very high **R² (98.7%)** → Model is performing excellently
* Very low MAE & RMSE → Predictions are close to actual values

---

## 📊 Actual vs Predicted Plot

```python
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(y_pred, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
```

---

## 🧪 Prediction System

Example input:

```python
input_data = (1447.160034, 78.370003, 15.2850, 1.474491)
```

Model output:

```
[85.55729996]
```

---

## 🏁 Conclusion

* Random Forest model predicts gold prices **very accurately**.
* High R² score proves strong predictive performance.
* Model captures patterns between Gold and other market indicators.

---

## 🔮 Future Enhancements

* Add **LSTM / ARIMA** for time-series forecasting
* Deploy model using **Flask / FastAPI**
* Create an interactive dashboard (Streamlit / Power BI)
* Perform advanced **hyperparameter tuning**

---



