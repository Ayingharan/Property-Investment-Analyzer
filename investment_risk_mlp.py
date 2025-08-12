import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class InvestmentRiskMLP:
    def __init__(self, data_path='prepared_property_dataset.csv'):
        self.data_path = data_path
        self.model_path = 'investment_risk_mlp_model.keras'
        self.scaler_path = 'investment_risk_mlp_scaler.pkl'
        self.model = None
        self.scaler = None

    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        X = dataset.drop(columns=['TAX_LEVY'], errors='ignore')
        y = dataset['TAX_LEVY']

        
        X = X.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' else col)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        joblib.dump(self.scaler, self.scaler_path)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, input_dim):
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        return model

    def run_complete_pipeline(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.model = self.build_model(X_train.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=256,
            callbacks=[early_stop],
            verbose=2
        )

        self.model.save(self.model_path)
        y_pred = self.model.predict(X_test).flatten()

        self.evaluate_model(y_test, y_pred)
        self.plot_predictions(y_test, y_pred)

        return True

    def evaluate_model(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print("\nInvestment Risk Model Evaluation Metrics (MLP):")
        print("--------------------------------------------------")
        print(f"R² Score               : {r2:.4f}")
        print(f"Mean Absolute Error    : {mae:.2f}")
        print(f"Mean Squared Error     : {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute % Error  : {mape:.2f}%")

        return {
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }

    def plot_predictions(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Actual TAX_LEVY")
        plt.ylabel("Predicted TAX_LEVY")
        plt.title("MLP Regressor: Actual vs Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("investment_mlp_predictions.png")
        plt.close()


if __name__ == '__main__':
    model = InvestmentRiskMLP()
    model.run_complete_pipeline()
