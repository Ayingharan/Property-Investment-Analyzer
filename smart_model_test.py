import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class SmartPropertyModelLoader:
    def __init__(self):
        self.models = {
            'tax_fairness': None,
            'zoning_classification': None,
            'investment_risk': None
        }
        self.scalers = {}
        self.label_encoder = None
        self.feature_selector = None

    def load_all_models(self):
        try:
            
            self.models['tax_fairness'] = tf.keras.models.load_model('_tax_fairness_model.keras')
            self.scalers['tax_fairness'] = joblib.load('_tax_model_scaler.pkl')

            
            self.models['zoning_classification'] = tf.keras.models.load_model('zoning_classifier.keras')
            self.scalers['zoning_classification'] = joblib.load('zoning_scaler.pkl')
            self.label_encoder = joblib.load('zoning_label_encoder.pkl')
            if os.path.exists('zoning_feature_selector.pkl'):
                self.feature_selector = joblib.load('zoning_feature_selector.pkl')

            
            self.models['investment_risk'] = tf.keras.models.load_model('investment_risk_mlp_model.keras')
            self.scalers['investment_risk'] = joblib.load('investment_risk_mlp_scaler.pkl')

            return True
        except Exception as e:
            print(f"[ModelLoader Error] {e}")
            return False


class PropertyPredictor:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def predict_all(self, df):
        results = {}

    
        current_year = pd.Timestamp.now().year
        if 'YEAR_BUILT' in df.columns and 'PROPERTY_AGE' not in df.columns:
            df['PROPERTY_AGE'] = current_year - df['YEAR_BUILT']

    
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]


        try:
            tax_X = df.drop(columns=['TAX_LEVY'], errors='ignore')
            expected_cols = self.model_loader.scalers['tax_fairness'].feature_names_in_
            tax_X = tax_X[expected_cols]
            tax_scaled = self.model_loader.scalers['tax_fairness'].transform(tax_X)
            tax_preds = self.model_loader.models['tax_fairness'].predict(tax_scaled).flatten()
            if 'TAX_LEVY' in df.columns:
                anomaly_count = np.sum(np.abs(tax_preds - df['TAX_LEVY']) > 5000)
            else:
                anomaly_count = 'N/A'
            results['tax_fairness'] = {
                'mean_prediction': float(np.mean(tax_preds)),
                'anomaly_count': int(anomaly_count) if anomaly_count != 'N/A' else anomaly_count
            }
        except Exception as e:
            results['tax_fairness'] = {'error': str(e)}


        try:
            zone_X = df.drop(columns=['ZONING'], errors='ignore')
            expected_cols = self.model_loader.scalers['zoning_classification'].feature_names_in_
            zone_X = zone_X[expected_cols]
            zone_scaled = self.model_loader.scalers['zoning_classification'].transform(zone_X)

            if self.model_loader.feature_selector:
                zone_scaled = self.model_loader.feature_selector.transform(zone_scaled)

            zone_preds = self.model_loader.models['zoning_classification'].predict(zone_scaled)
            confidences = np.max(zone_preds, axis=1)
            low_conf = np.sum(confidences < 0.6)
            results['zoning_classification'] = {
                'average_confidence': float(np.mean(confidences)),
                'low_confidence_count': int(low_conf)
            }
        except Exception as e:
            results['zoning_classification'] = {'error': str(e)}


        try:
            risk_X = df.drop(columns=['TAX_LEVY'], errors='ignore')
            expected_cols = self.model_loader.scalers['investment_risk'].feature_names_in_
            risk_X = risk_X[expected_cols]
            risk_scaled = self.model_loader.scalers['investment_risk'].transform(risk_X)
            risk_preds = self.model_loader.models['investment_risk'].predict(risk_scaled).flatten()
            high_risk = np.sum(risk_preds > 0.75)
            results['investment_risk'] = {
                'mean_risk_score': float(np.mean(risk_preds)),
                'high_risk_count': int(high_risk)
            }
        except Exception as e:
            results['investment_risk'] = {'error': str(e)}

        return results


if __name__ == '__main__':
    print("[DEBUG] Running model loader test...")
    loader = SmartPropertyModelLoader()
    if loader.load_all_models():
        print("[SUCCESS] All models loaded successfully.")
    else:
        print("[ERROR] Failed to load one or more models.")

    
    if os.path.exists('prepared_property_dataset.csv'):
        sample_df = pd.read_csv('prepared_property_dataset.csv').head(10)
        predictor = PropertyPredictor(loader)
        result = predictor.predict_all(sample_df)
        print("[DEBUG] Sample prediction output:")
        print(result)
    else:
        print("[WARNING] No sample data file found to test predictions.")
