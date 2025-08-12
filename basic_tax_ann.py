

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
tf.random.set_seed(42)

class BasicTaxFairnessANN:
    """
    Basic Artificial Neural Network for Tax Fairness Prediction
    
    This model uses a simple architecture without regularization
    to predict fair tax assessments based on property characteristics.
    """
    
    def __init__(self, data_path='prepared_property_dataset.csv'):
        """Initialize the Basic Tax Fairness ANN"""
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None
        self.predictions = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print(" Loading and preparing data...")
        
        try:
            
            data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully. Shape: {data.shape}")
            
            
            if 'TAX_LEVY' not in data.columns:
                print(" Error: TAX_LEVY column not found!")
                return False
            
            
            X = data.drop(columns=['TAX_LEVY'])
            y = data['TAX_LEVY']
            
            
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = pd.factorize(X[col])[0]
            
            
            X = X.fillna(X.median())
            y = y.fillna(y.median())
            
            
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]
            
            print(f"Data after outlier removal: {len(X)} samples")
            
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print(f" Data preparation completed:")
            print(f"   Training samples: {self.X_train_scaled.shape[0]}")
            print(f"   Test samples: {self.X_test_scaled.shape[0]}")
            print(f"   Features: {self.X_train_scaled.shape[1]}")
            
            return True
            
        except Exception as e:
            print(f" Error in data preparation: {str(e)}")
            return False
    
    def build_model(self):
        """Build a basic ANN model"""
        print("\n Building Basic ANN Model...")
        
        input_dim = self.X_train_scaled.shape[1]
        
        
        model = Sequential([
            
            Dense(64, input_dim=input_dim, activation='relu'),
            
            
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            
            
            Dense(1, activation='linear')
        ])
        
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print(" Basic ANN Model Architecture:")
        model.summary()
        print(f"Total Parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, epochs=100, batch_size=32, verbose=1):
        """Train the basic ANN model"""
        print("\n Training Basic ANN Model...")
        
        if self.X_train_scaled is None:
            print(" Data not prepared. Please run load_and_prepare_data() first.")
            return False
        
        
        self.model = self.build_model()
        
        
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose,
            shuffle=True
        )
        
        print(" Training completed!")
        return True
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n Evaluating Model Performance...")
        
        if self.model is None:
            print(" Model not trained. Please run train_model() first.")
            return None
        
        
        self.predictions = self.model.predict(self.X_test_scaled, verbose=0)
        self.predictions = self.predictions.flatten()
        
        
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)
        
        
        tolerance = 0.1
        accurate_predictions = np.abs(self.y_test - self.predictions) / self.y_test <= tolerance
        accuracy_percentage = np.mean(accurate_predictions) * 100
        
        
        print("="*50)
        print("BASIC ANN MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Mean Squared Error (MSE):     {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE):    {mae:.2f}")
        print(f"R² Score:                     {r2:.4f}")
        print(f"Predictions within 10% tolerance: {accuracy_percentage:.2f}%")
        print("="*50)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy_percentage': accuracy_percentage
        }
    
    def visualize_results(self):
        """Create visualization of model performance"""
        print("\n Creating Performance Visualizations...")
        
        if self.history is None or self.predictions is None:
            print(" Model not trained or evaluated. Please train and evaluate first.")
            return
        
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Basic ANN Tax Fairness Model - Performance Analysis', 
                    fontsize=16, fontweight='bold')
        
        
        ax1 = axes[0, 0]
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        ax1.set_title('Model Training History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        ax2 = axes[0, 1]
        ax2.scatter(self.y_test, self.predictions, alpha=0.6, s=30)
        min_val = min(self.y_test.min(), self.predictions.min())
        max_val = max(self.y_test.max(), self.predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Tax Levy')
        ax2.set_ylabel('Predicted Tax Levy')
        ax2.set_title('Actual vs Predicted Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        
        ax3 = axes[1, 0]
        residuals = self.y_test - self.predictions
        ax3.scatter(self.predictions, residuals, alpha=0.6, s=30, color='green')
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Tax Levy')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals Plot')
        ax3.grid(True, alpha=0.3)
        
        
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title('Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        
        plt.savefig('basic_tax_ann_evaluation.png', dpi=300, bbox_inches='tight')
        print(" Plots saved as 'basic_tax_ann_evaluation.png'")
    
    def detect_anomalies(self, threshold_percentile=95):
        """Detect anomalous tax assessments"""
        print("\n Detecting Tax Assessment Anomalies...")
        
        if self.predictions is None:
            print(" Model not evaluated. Please run evaluate_model() first.")
            return None
        
        
        absolute_errors = np.abs(self.y_test - self.predictions)
        percentage_errors = (absolute_errors / self.y_test) * 100
        
        
        threshold = np.percentile(percentage_errors, threshold_percentile)
        
        
        anomalies_mask = percentage_errors > threshold
        num_anomalies = np.sum(anomalies_mask)
        
        print(f"Anomaly Detection Results:")
        print(f"  Threshold (top {100-threshold_percentile}%): {threshold:.2f}%")
        print(f"  Number of anomalies detected: {num_anomalies}")
        print(f"  Anomaly rate: {(num_anomalies/len(self.predictions))*100:.2f}%")
        
        if num_anomalies > 0:
            print(f"\nTop 5 Most Anomalous Properties:")
            anomaly_indices = np.where(anomalies_mask)[0]
            top_anomalies = anomaly_indices[np.argsort(percentage_errors[anomalies_mask])[-5:]]
            
            for i, idx in enumerate(top_anomalies):
                actual = self.y_test.iloc[idx]
                predicted = self.predictions[idx]
                error = percentage_errors.iloc[idx]
                print(f"  {i+1}. Index {idx}: Actual=${actual:.2f}, Predicted=${predicted:.2f}, Error={error:.1f}%")
        
        return {
            'anomaly_indices': np.where(anomalies_mask)[0],
            'percentage_errors': percentage_errors[anomalies_mask],
            'threshold': threshold,
            'anomaly_rate': (num_anomalies/len(self.predictions))*100
        }
    
    def save_model(self, model_path='basic_tax_fairness_model.keras'):
        """Save the trained model"""
        print(f"\n Saving Model...")
        
        if self.model is None:
            print(" No model to save. Please train the model first.")
            return False
        
        try:
            
            self.model.save(model_path)
            
            
            import joblib
            joblib.dump(self.scaler, 'basic_tax_scaler.pkl')
            
            
            metadata = {
                'model_type': 'Basic Tax Fairness ANN',
                'input_features': self.X_train_scaled.shape[1],
                'training_samples': len(self.X_train_scaled),
                'test_samples': len(self.X_test_scaled),
                'model_path': model_path,
                'scaler_path': 'basic_tax_scaler.pkl'
            }
            joblib.dump(metadata, 'basic_tax_metadata.pkl')
            
            print(f" Model saved successfully:")
            print(f"   Model: {model_path}")
            print(f"   Scaler: basic_tax_scaler.pkl")
            print(f"   Metadata: basic_tax_metadata.pkl")
            
            return True
            
        except Exception as e:
            print(f" Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path='basic_tax_fairness_model.keras'):
        """Load a saved model"""
        print(f"\n Loading Model...")
        
        try:
            
            self.model = tf.keras.models.load_model(model_path)
            
            
            import joblib
            self.scaler = joblib.load('basic_tax_scaler.pkl')
            
            
            metadata = joblib.load('basic_tax_metadata.pkl')
            
            print(f" Model loaded successfully:")
            print(f"   Model type: {metadata['model_type']}")
            print(f"   Input features: {metadata['input_features']}")
            
            return True
            
        except Exception as e:
            print(f" Error loading model: {str(e)}")
            return False
    
    def predict_single_property(self, property_data):
        """Make prediction for a single property"""
        if self.model is None:
            print(" Model not loaded. Please train or load a model first.")
            return None
        
        try:
            
            if isinstance(property_data, dict):
                property_data = pd.DataFrame([property_data])
            elif isinstance(property_data, list):
                property_data = np.array(property_data).reshape(1, -1)
            
            
            if hasattr(property_data, 'values'):
                scaled_data = self.scaler.transform(property_data.values)
            else:
                scaled_data = self.scaler.transform(property_data)
            
            
            prediction = self.model.predict(scaled_data, verbose=0)[0][0]
            
            return float(prediction)
            
        except Exception as e:
            print(f" Error making prediction: {str(e)}")
            return None
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        print(" BASIC TAX FAIRNESS ANN - COMPLETE PIPELINE")
        print("="*60)
        
        
        if not self.load_and_prepare_data():
            print(" Pipeline failed at data preparation")
            return False
        
        
        if not self.train_model(epochs=100, batch_size=32):
            print(" Pipeline failed at model training")
            return False
        
        
        results = self.evaluate_model()
        if results is None:
            print(" Pipeline failed at model evaluation")
            return False
        
        
        self.visualize_results()
        
        
        anomalies = self.detect_anomalies()
        
        
        if not self.save_model():
            print(" Pipeline failed at model saving")
            return False
        
        print("\n" + "="*60)
        print(" BASIC TAX FAIRNESS ANN PIPELINE COMPLETED!")
        print("="*60)
        print("\n Final Results Summary:")
        print(f"   R² Score: {results['r2']:.4f}")
        print(f"   RMSE: {results['rmse']:.2f}")
        print(f"   Accuracy (10% tolerance): {results['accuracy_percentage']:.2f}%")
        if anomalies:
            print(f"   Anomalies detected: {len(anomalies['anomaly_indices'])}")
        
        return True


def load_and_predict(model_path='basic_tax_fairness_model.keras', property_data=None):
    """Utility function to load model and make predictions"""
    model = BasicTaxFairnessANN()
    
    if model.load_model(model_path):
        if property_data is not None:
            prediction = model.predict_single_property(property_data)
            print(f"Predicted Tax Levy: ${prediction:.2f}")
            return prediction
        else:
            print(" Model loaded successfully. Ready for predictions.")
            return model
    else:
        print(" Failed to load model")
        return None

def compare_properties(model_path='basic_tax_fairness_model.keras', properties_list=None):
    """Compare tax predictions for multiple properties"""
    model = BasicTaxFairnessANN()
    
    if not model.load_model(model_path):
        print(" Failed to load model")
        return None
    
    if properties_list is None:
        print(" No properties provided for comparison")
        return None
    
    results = []
    for i, property_data in enumerate(properties_list):
        prediction = model.predict_single_property(property_data)
        results.append({
            'property_id': i+1,
            'predicted_tax': prediction,
            'property_data': property_data
        })
        print(f"Property {i+1}: Predicted Tax = ${prediction:.2f}")
    
    return results


if __name__ == "__main__":
    print(" BASIC TAX FAIRNESS ANN MODEL")
    print("="*50)
    print("This is a simple neural network for tax fairness prediction")
    print("without regularization techniques.")
    print("="*50)
    
    
    model = BasicTaxFairnessANN()
    success = model.run_complete_pipeline()
    
    if success:
        print("\n Basic Tax Fairness ANN completed successfully!")
        print("\n Key Features of this Basic Model:")
        print("   • Simple 3-layer architecture")
        print("   • No regularization (prone to overfitting)")
        print("   • Fast training and inference")
        print("   • Good baseline performance")
        print("   • Easy to understand and implement")
        
        print("\n Model Characteristics:")
        print("   • Architecture: 64 → 32 → 16 → 1 neurons")
        print("   • Activation: ReLU (hidden), Linear (output)")
        print("   • Optimizer: Adam")
        print("   • Loss: Mean Squared Error")
        print("   • No dropout or regularization")
        
        print("\n Next Steps:")
        print("   • Compare with regularized model")
        print("   • Experiment with different architectures")
        print("   • Add regularization techniques")
        print("   • Optimize hyperparameters")
    else:
        print("\n Basic Tax Fairness ANN pipeline failed!")
        print("Please check the error messages above.")