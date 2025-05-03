

# 🩺 Diabetes Prediction with Machine Learning

This project applies a variety of machine learning algorithms to predict whether a patient has diabetes based on medical measurements. It covers **EDA**, **model training**, **hyperparameter tuning**, and **making real-time predictions**.

---

## 📊 Project Overview

The notebook explores multiple supervised learning methods to classify diabetes using the Pima Indians Diabetes dataset.

### Key Stages:
- ✅ Exploratory Data Analysis (EDA)
- 🧼 Outlier Removal
- 🧪 Model Training and Testing
- 📈 Evaluation Metrics
- 🔍 Hyperparameter Tuning using `GridSearchCV`
- 🔮 Real-Time Prediction

---

## 🧠 Algorithms Used

| Category                     | Models                              |
|-----------------------------|-------------------------------------|
| Distance / Gradient-Based   | Logistic Regression, KNN, SVC       |
| Probability / Tree-Based    | Naive Bayes, Decision Tree, Random Forest, XGBoost |

---

## 🔍 Highlight: XGBoost with GridSearchCV

A fine-tuned XGBoost model was trained using `GridSearchCV` to find the optimal combination of:
- `learning_rate`
- `n_estimators`
- `max_depth`

```python
Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}
Best cross-validation score: 0.805
```

---

## 🧪 Sample Predictions

```python
# Predicting for two sample patients
print('1st Prediction:', gs1.predict([[0, 120, 60, 18, 106, 25, 0.717, 23]]))  # Output: 0
print('2nd Prediction:', gs1.predict([[0, 140, 60, 18, 106, 30, 0.717, 40]]))  # Output: 1
```

> The model predicts that the **first patient** is **not diabetic**, while the **second patient** is — likely due to elevated glucose and age values.

---


Main libraries used:
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`

---

## 🚀 Run the Notebook

```bash
jupyter notebook diabetesPred.ipynb
```

