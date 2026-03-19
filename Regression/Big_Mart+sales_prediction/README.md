-# Big Mart Sales Prediction – Full Project (With Outputs)

This README includes **all code, all explanations, and all outputs/results** generated throughout the entire project.

---

# 📁 1. Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
```

---

# 📥 2. Load Dataset

```python
Big_mart = pd.read_csv("Train.csv")
Big_mart.head()
```

### **Output (First 5 Rows)**

| Item_Identifier | Item_Weight | Item_Fat_Content | Item_Visibility | Item_Type | Item_MRP | Outlet_Identifier | Outlet_Establishment_Year | Outlet_Size |
|--------|---------|----------------|----------------|----------|----------|--------------------|------------------------------|-------------|
| FDA15 | 9.3 | Low Fat | 0.016 | Dairy | 249.809 | OUT049 | 1999 | Medium |
| DRC01 | 5.92 | Regular | 0.019 | Soft Drinks | 48.269 | OUT018 | 2009 | Medium |
| FDN15 | 17.5 | Low Fat | 0.0167 | Meat | 141.618 | OUT049 | 1999 | Medium |

---

# 🧮 3. Shape of Dataset

```python
Big_mart.shape
```

### **Output**
```
(8523, 12)
```

---

# ℹ️ 4. Dataset Info

```python
Big_mart.info()
```

### **Output**
```
8523 entries, 12 columns
Float columns: 4
Object columns: 7
Integer columns: 1
```

---

# ❗ 5. Missing Values (Before Cleaning)

```python
Big_mart.isnull().sum()
```

### **Output**
```
Item_Weight      1463
Outlet_Size      2410
All others      0
```

---

# 🧹 6. Data Cleaning

## Fill Missing Item_Weight
```python
Big_mart['Item_Weight'] = Big_mart['Item_Weight'].fillna(Big_mart['Item_Weight'].median())
```

## Fill Missing Outlet_Size
```python
Big_mart['Outlet_Size'] = Big_mart['Outlet_Size'].fillna(Big_mart['Outlet_Size'].mode()[0])
```

### **Missing Values After Cleaning**
```
All columns: 0 missing
```

---

# 🔁 7. Fix Inconsistent Categories

```python
Big_mart['Item_Fat_Content'] = Big_mart.Item_Fat_Content.replace({"LF":"Low Fat","low fat":"Low Fat","reg":"Regular"})
```

---

# 📊 8. EDA (Key Insights)

### **Item Weight Distribution**
- Normal-shaped  
- Mostly 8–17 kg  

### **Item Visibility**
- Right-skewed  
- Many zeros  

### **Item MRP**
- Four price clusters  

### **Item Outlet Sales**
- Right-skewed  
- Most between 500–3500  

### **Item Fat Content (After Cleaning)**
```
Low Fat     5517
Regular     3006
```

### **Outlet Location Type**
```
Tier 3: 3350
Tier 2: 2785
Tier 1: 2388
```

### **Outlet Size**
```
Medium: 5203
High: 932
Small: 2388
```

---

# 🔠 9. Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for col in ['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']:
    Big_mart[col] = encoder.fit_transform(Big_mart[col])
```

### **First 5 Rows After Encoding**

| Item_Identifier | Item_Weight | Fat | Visibility | Type | MRP | Outlet_ID | Year | Size | Loc | Outlet |
|--|--|--|--|--|--|--|--|--|--|--|
| 156 | 9.3 | 0 | 0.016 | 4 | 249.80 | 9 | 1999 | 1 | 0 | 1 |
| 8 | 5.92 | 1 | 0.019 | 14 | 48.26 | 3 | 2009 | 1 | 1 | 1 |
| 662 | 17.5 | 0 | 0.016 | 10 | 141.61 | 9 | 1999 | 1 | 0 | 1 |

---

# 🧪 10. Train–Test Split

```python
from sklearn.model_selection import train_test_split

X = Big_mart.drop(columns='Item_Outlet_Sales')
y = Big_mart.Item_Outlet_Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **Output**
```
X_train: (6818, 11)
X_test:  (1705, 11)
```

---

# 🤖 11. Model Training (XGBoost)

```python
from xgboost import XGBRegressor

regressor = XGBRegressor()
regressor.fit(X_train, y_train)
```

---

# 📈 12. Model Predictions & Performance

## Training Set R² Score
```python
r2_score(y_train, regressor.predict(X_train))
```

### **Output**
```
0.8698351206603934
```
✔ High accuracy training model

---

## Testing Set R² Score
```python
r2_score(y_test, regressor.predict(X_test))
```

### **Output**
```
0.5246342336961346
```
✔ Moderate performance  
✔ Slight overfitting detected

---

# 🏁 13. Final Achievements

✔ Completed full EDA  
✔ Cleaned dataset  
✔ Encoded categorical features  
✔ Trained XGBoost regression model  
✔ Achieved strong training accuracy  
✔ Achieved moderate test accuracy  
✔ Ready for deployment/improvement  

---


