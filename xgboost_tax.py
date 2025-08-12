
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('prepared_property_dataset.csv')
target_column = 'TAX_LEVY'
X = dataset.drop(columns=[target_column])
y = dataset[target_column]


X = X.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' else col)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n Tax Fairness Model (XGBoost) Evaluation")
print("---------------------------------------------")
print(f" R² Score           : {r2:.4f}")
print(f" MAE                : {mae:.2f}")
print(f" MSE                : {mse:.2f}")
print(f" RMSE               : {rmse:.2f}")
print(f" MAPE               : {mape:.2f}%")


joblib.dump(model, 'tax_fairness_xgb_model.pkl')
joblib.dump(scaler, 'tax_fairness_xgb_scaler.pkl')


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual TAX_LEVY")
plt.ylabel("Predicted TAX_LEVY")
plt.title("XGBoost Regressor: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("tax_fairness_xgboost_predictions.png")
plt.show()
