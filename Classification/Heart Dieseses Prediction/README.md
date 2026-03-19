# ❤️ Heart Disease Prediction Using Machine Learning

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/82132e06-1b65-4d92-992c-edcd9b685c05" />


## 📌 Project Overview

This project predicts whether a person has **heart disease** using
machine learning, specifically **Logistic Regression**.
The dataset contains patient health measurements such as age,
cholesterol level, chest pain type, blood pressure, and more.

The workflow includes: - Importing libraries
- Loading the dataset
- Understanding the data dictionary
- Exploratory data analysis
- Splitting data into train/test
- Building a Logistic Regression model
- Evaluating model performance
- Visualizing the confusion matrix

------------------------------------------------------------------------

## 📚 Libraries Used

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
```

## 📥 Dataset Information

**Dataset:** heart.csv
**Rows:** 303
**Columns:** 14

------------------------------------------------------------------------

## 📊 Dataset Sample

    age sex cp  trestbps    chol    fbs restecg thalach exang   oldpeak slope   ca  thal    target
    63  1   3   145 233 1   0   150 0   2.3 0   0   1   1
    37  1   2   130 250 0   1   187 0   3.5 0   0   2   1
    41  0   1   130 204 0   0   172 0   1.4 2   0   2   1
    56  1   1   120 236 0   1   178 0   0.8 2   0   2   1
    57  0   0   120 354 0   1   163 1   0.6 2   0   2   1

------------------------------------------------------------------------

## 📖 Data Dictionary

  Feature    Description
  ---------- -----------------------------------
  age        Patient age
  sex        1 = male, 0 = female
  cp         Chest pain type (0--3)
  trestbps   Resting blood pressure
  chol       Serum cholesterol (mg/dl)
  fbs        Fasting blood sugar \> 120 mg/dl
  restecg    ECG results
  thalach    Maximum heart rate achieved
  exang      Exercise-induced angina
  oldpeak    ST depression induced by exercise
  slope      Slope of ST segment
  ca         Major vessels colored
  thal       Heart condition type
  target     0 = no disease, 1 = disease

------------------------------------------------------------------------

## 📏 Dataset Shape

    (303, 14)

------------------------------------------------------------------------

## 🧪 Missing Values

    0 missing values

------------------------------------------------------------------------

## 📊 Outcome Distribution

    1 (Heart Disease): 54.45%
    0 (No Disease): 45.54%

------------------------------------------------------------------------

## 🔍 Exploratory Data Analysis

### Countplot of Target

``` python
sns.countplot(x='target', data=df, palette='hls')
```

### Age Distribution

``` python
sns.boxplot(data=df, x='age')
```

------------------------------------------------------------------------

## 🛠 Data Preprocessing

### Splitting Features and Target

``` python
X = df.drop('target', axis=1)
y = df['target']
```

### Train-Test Split

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
```

**Train Shapes**

    X_train: (242, 13)
    y_train: (242,)

**Test Shapes**

    X_test: (61, 13)
    y_test: (61,)

------------------------------------------------------------------------

## 🤖 Model Training --- Logistic Regression

``` python
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

------------------------------------------------------------------------

## 📈 Evaluation Metrics

### Confusion Matrix

``` python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
```

### Classification Report

``` python
print(classification_report(y_test, y_pred))
```

------------------------------------------------------------------------

## ✔ Results

  Metric                       Score
  ---------------------------- -------
  **Accuracy**                 0.77
  **Precision (No Disease)**   0.83
  **Recall (No Disease)**      0.67
  **Precision (Disease)**      0.73
  **Recall (Disease)**         0.87

------------------------------------------------------------------------

## 📌 Conclusion

The Logistic Regression model achieved **77% accuracy**.

It performs better at **detecting heart disease (target = 1)** than
detecting normal cases.

This project demonstrates a complete ML workflow:\
**load → explore → preprocess → train → evaluate → visualize**

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Try Random Forest, SVM, XGBoost
-   Hyperparameter tuning
-   Add more visualizations
