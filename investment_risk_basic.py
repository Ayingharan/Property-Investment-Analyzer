import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class InvestmentRiskANN:
    def __init__(self, dataset_path='prepared_property_dataset.csv'):
        self.dataset_path = dataset_path
        self.model_path = 'investment_risk_basic.keras'
        self.scaler_path = 'investment_risk_basic_scaler.pkl'

    def prepare_data(self):
        df = pd.read_csv(self.dataset_path)
        X = df.drop(columns=['TAX_LEVY'])
        y = df['TAX_LEVY']
        X = X.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' else col)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, self.scaler_path)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        return model


    def run_complete_pipeline(self):
        print(" Running Investment Risk Basic Pipeline...")

        X_train, X_test, y_train, y_test = self.prepare_data()

        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
            print(" Loaded existing model. Skipping training.")
        else:
            model = self.build_model(X_train.shape[1])
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=512, callbacks=[early_stop], verbose=2)
            model.save(self.model_path)
            print("Model trained and saved.")

        y_pred = model.predict(X_test).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\nInvestment Risk Basic Evaluation:")
        print("----------------------------------------")
        print(f"R² Score               : {r2:.4f}")
        print(f"Mean Absolute Error    : {mae:.2f}")
        print(f"Mean Squared Error     : {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.xlabel("Actual TAX_LEVY")
        plt.ylabel("Predicted TAX_LEVY")
        plt.title("Actual vs Predicted Investment Risk")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("investment_risk_basic_predictions.png")
        plt.close()

if __name__ == '__main__':
    ann = InvestmentRiskANN()
    ann.run_complete_pipeline()
