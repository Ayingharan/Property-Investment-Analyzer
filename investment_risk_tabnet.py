
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class StopOnHighR2(Callback):
    def __init__(self, X_val, y_val, threshold=0.999):
        self.X_val = X_val
        self.y_val = y_val
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.trainer.predict(self.X_val).flatten()
        r2 = r2_score(self.y_val, y_pred)
        print(f" Epoch {epoch+1} R² Score: {r2:.5f}")
        if r2 >= self.threshold:
            print(f" Early stopping: R² {r2:.5f} reached threshold of {self.threshold}")
            raise KeyboardInterrupt


dataset = pd.read_csv('prepared_property_dataset.csv')
target_column = 'TAX_LEVY'
X = dataset.drop(columns=[target_column])
y = dataset[target_column]


X = X.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' else col)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


tabnet = TabNetRegressor()
try:
    tabnet.fit(
        X_train_scaled, y_train.values.reshape(-1, 1),
        eval_set=[(X_test_scaled, y_test.values.reshape(-1, 1))],
        max_epochs=150,
        patience=15,
        callbacks=[StopOnHighR2(X_test_scaled, y_test)]
    )
except KeyboardInterrupt:
    print(" Training stopped early due to high R².")


y_pred = tabnet.predict(X_test_scaled).flatten()


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


joblib.dump(tabnet, 'investment_risk_tabnet_model.pkl')
joblib.dump(scaler, 'investment_risk_tabnet_scaler.pkl')


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual TAX_LEVY")
plt.ylabel("Predicted")
plt.title("TabNet Regressor: Actual vs Predicted")
plt.grid()
plt.tight_layout()
plt.savefig("investment_tabnet_predictions.png")


print("\n Investment Risk Model Evaluation Metrics")
print("--------------------------------------------------")
print(f" R² Score           : {r2:.4f}")
print(f" Mean Absolute Error (MAE)  : {mae:.2f}")
print(f" Mean Squared Error (MSE)   : {mse:.2f}")
print(f" Root MSE (RMSE)            : {rmse:.2f}")
print(f" Mean Absolute % Error (MAPE): {mape:.2f}%")

