import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# Загружаем данные

data = pd.read_excel('Объёмы перевозок.xls')
print(data)

data2 = pd.read_excel('МС_Владимирская область.xls')
print(data2)


# Разделяем признаки и целевую переменную
X = data.filter(['2208'])
print(X)
plt.plot(data)
plt.show()
y = data['2310']

# Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабируем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Логистическая регрессия
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Случайный лес
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# XGBoost
xg_model = XGBClassifier(random_state=42)
xg_model.fit(X_train_scaled, y_train)
xg_pred = xg_model.predict(X_test_scaled)

# Оценка моделей
models = [('Logistic Regression', lr_pred),
          ('Random Forest', rf_pred),
          ('XGBoost', xg_pred)]

metrics = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

for model_name, predictions in models:
    print(f'\n{model_name}:')
    for metric, metric_name in zip(metrics, metric_names):
        score = metric(y_test, predictions)
        print(f'{metric_name}: {score:.4f}')
