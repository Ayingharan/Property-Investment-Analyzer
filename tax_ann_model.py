

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import IsolationForest
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import joblib
import logging
import os
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaxFairnessANN:
    def __init__(self, data_path='prepared_property_dataset.csv'):
        """Initialize the  Tax Fairness ANN model"""
        self.data_path = data_path
        self.model = None
        self.scaler = RobustScaler()
        self.target_scaler = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_val_scaled = None
        self.y_train_scaled = None
        self.y_test_scaled = None
        self.y_val_scaled = None
        self.history = None
        self.predictions = None
        self.dataset = None
        self.feature_columns = None
        
    def load_and_prepare_data(self):
        """ data loading and preparation with advanced preprocessing"""
        try:
            
            self.dataset = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.dataset.shape}")
            
            
            if 'TAX_LEVY' not in self.dataset.columns:
                logger.error("Target variable 'TAX_LEVY' not found in dataset")
                return False
            
            
            logger.info("\n=== DATA EXPLORATION ===")
            logger.info(f"Dataset shape: {self.dataset.shape}")
            logger.info(f"Missing values: {self.dataset.isnull().sum().sum()}")
            logger.info(f"TAX_LEVY stats:\n{self.dataset['TAX_LEVY'].describe()}")
            
            
            original_size = len(self.dataset)
            
            
            self.dataset = self.dataset[self.dataset['TAX_LEVY'] > 0]
            self.dataset = self.dataset[self.dataset['TAX_LEVY'].notna()]
            
            
            Q1 = self.dataset['TAX_LEVY'].quantile(0.01)
            Q99 = self.dataset['TAX_LEVY'].quantile(0.99)
            self.dataset = self.dataset[
                (self.dataset['TAX_LEVY'] >= Q1) &
                (self.dataset['TAX_LEVY'] <= Q99)
            ]
            
            logger.info(f"Removed {original_size - len(self.dataset)} problematic records")
            logger.info(f"Final dataset shape: {self.dataset.shape}")
            
            
            X = self.dataset.drop(columns=['TAX_LEVY'])
            y = self.dataset['TAX_LEVY']
            
            
            logger.info("\n=== FEATURE PREPROCESSING ===")
            
            
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                
                if X[col].nunique() <= 50:
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:
                    X[col] = pd.factorize(X[col].astype(str))[0]
            
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isnull().sum() > 0:
                    
                    X[col] = X[col].fillna(X[col].median())
            
            
            logger.info("Creating derived features...")
            
            
            if 'TOTAL_AREA' in X.columns and 'ROOMS' in X.columns:
                X['AREA_PER_ROOM'] = X['TOTAL_AREA'] / (X['ROOMS'] + 1)
            
            
            if 'YEAR_BUILT' in X.columns:
                current_year = pd.Timestamp.now().year
                X['PROPERTY_AGE'] = current_year - X['YEAR_BUILT']
            
            
            constant_features = X.columns[X.nunique() <= 1]
            if len(constant_features) > 0:
                X = X.drop(constant_features, axis=1)
                logger.info(f"Removed {len(constant_features)} constant features")
            
            
            self.feature_columns = X.columns.tolist()
            logger.info(f"Final feature count: {len(self.feature_columns)}")
            
            
            logger.info("Detecting multivariate outliers...")
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            
            inlier_mask = outlier_labels == 1
            X = X[inlier_mask]
            y = y[inlier_mask]
            
            logger.info(f"Removed {np.sum(~inlier_mask)} multivariate outliers")
            logger.info(f"Final dataset for training: {X.shape}")
            
            
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=None
            )
            
            
            logger.info("\n=== FEATURE SCALING ===")
            
            
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_val_scaled = self.scaler.transform(self.X_val)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            
            if y.std() > 10000:
                logger.info("Applying target scaling for training stability")
                self.target_scaler = RobustScaler()
                self.y_train_scaled = self.target_scaler.fit_transform(
                    self.y_train.values.reshape(-1, 1)
                ).flatten()
                self.y_val_scaled = self.target_scaler.transform(
                    self.y_val.values.reshape(-1, 1)
                ).flatten()
                self.y_test_scaled = self.target_scaler.transform(
                    self.y_test.values.reshape(-1, 1)
                ).flatten()
            else:
                self.y_train_scaled = self.y_train.values
                self.y_val_scaled = self.y_val.values
                self.y_test_scaled = self.y_test.values
            
            logger.info(f"Training set shape: {self.X_train_scaled.shape}")
            logger.info(f"Validation set shape: {self.X_val_scaled.shape}")
            logger.info(f"Test set shape: {self.X_test_scaled.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            return False
    
    def build_model(self, input_dim):
        """Build an  ANN model with advanced architecture"""
        logger.info("Building  model architecture...")
        
        model = Sequential([
            
            Dense(256, input_dim=input_dim, 
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            Dropout(0.3),
            
            
            Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            Dropout(0.3),
            
            Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            Dropout(0.2),
            
            Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            Dropout(0.2),
            
            Dense(16, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            tf.keras.layers.Activation('relu'),
            Dropout(0.1),
            
            
            Dense(1, activation='linear')
        ])
        
        
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, epochs=300, batch_size=64):
        """ training with better strategy and callbacks"""
        if self.X_train_scaled is None:
            logger.error("Data not prepared. Call load_and_prepare_data() first.")
            return False
        
        
        self.model = self.build_model(self.X_train_scaled.shape[1])
        
        
        logger.info("\n Model Architecture:")
        self.model.summary()
        
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1,
                cooldown=5
            ),
            ModelCheckpoint(
                'best_improved_tax_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        
        logger.info("Starting  model training...")
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train_scaled,
            validation_data=(self.X_val_scaled, self.y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        logger.info(" model training completed!")
        return True
    
    def evaluate_model(self):
        """Comprehensive model evaluation with  metrics"""
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return False
        
        
        predictions_scaled = self.model.predict(self.X_test_scaled, verbose=0)
        
        
        if self.target_scaler is not None:
            self.predictions = self.target_scaler.inverse_transform(
                predictions_scaled
            ).flatten()
            y_test_original = self.target_scaler.inverse_transform(
                self.y_test_scaled.reshape(-1, 1)
            ).flatten()
        else:
            self.predictions = predictions_scaled.flatten()
            y_test_original = self.y_test_scaled
        
        
        mse = mean_squared_error(y_test_original, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, self.predictions)
        r2 = r2_score(y_test_original, self.predictions)
        
        
        mask = y_test_original != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test_original[mask] - self.predictions[mask]) / y_test_original[mask])) * 100
        else:
            mape = float('inf')
        
        
        median_ae = np.median(np.abs(y_test_original - self.predictions))
        max_error = np.max(np.abs(y_test_original - self.predictions))
        
        
        tolerances = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20%
        accuracy_results = {}
        
        for tol in tolerances:
            accurate_predictions = np.abs(y_test_original - self.predictions) <= tol * y_test_original
            accuracy_results[f'within_{int(tol*100)}pct'] = np.mean(accurate_predictions) * 100
        
        
        logger.info("\n" + "="*60)
        logger.info(" MODEL EVALUATION METRICS")
        logger.info("="*60)
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logger.info(f"Median Absolute Error: {median_ae:.2f}")
        logger.info(f"Maximum Error: {max_error:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        logger.info("")
        logger.info("PREDICTION ACCURACY BY TOLERANCE:")
        for tol, acc in accuracy_results.items():
            logger.info(f"  Predictions {tol}: {acc:.2f}%")
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'median_ae': median_ae,
            'max_error': max_error, 'r2': r2, 'mape': mape,
            **accuracy_results
        }
    
    def visualize_results(self):
        """ visualization with comprehensive plots"""
        if self.history is None or self.predictions is None:
            logger.error("Model not trained or evaluated. Train and evaluate first.")
            return
        
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        
        if self.target_scaler is not None:
            y_test_original = self.target_scaler.inverse_transform(
                self.y_test_scaled.reshape(-1, 1)
            ).flatten()
        else:
            y_test_original = self.y_test_scaled
        
        
        plt.subplot(3, 3, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        plt.title(' Model Training History', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Huber)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        
        plt.subplot(3, 3, 2)
        if 'lr' in self.history.history:
            plt.plot(self.history.history['lr'], linewidth=2, color='green')
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 3)
        plt.scatter(y_test_original, self.predictions, alpha=0.6, s=30, color='blue')
        min_val = min(y_test_original.min(), self.predictions.min())
        max_val = max(y_test_original.max(), self.predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual TAX_LEVY')
        plt.ylabel('Predicted TAX_LEVY')
        plt.title('Actual vs Predicted TAX_LEVY', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 4)
        residuals = y_test_original - self.predictions
        plt.scatter(self.predictions, residuals, alpha=0.6, s=30, color='purple')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted TAX_LEVY')
        plt.ylabel('Residuals')
        plt.title(' Residuals Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 5)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Residuals)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 6)
        plt.hist(residuals, bins=40, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Error Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 7)
        percentage_errors = []
        for actual, pred in zip(y_test_original, self.predictions):
            if actual != 0 and abs(actual) > 1e-6:
                pct_error = abs((pred - actual) / actual * 100)
                if pct_error < 500:
                    percentage_errors.append(pct_error)
        
        if percentage_errors:
            plt.hist(percentage_errors, bins=40, density=True, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            plt.xlabel('Absolute Percentage Error (%)')
            plt.ylabel('Density')
            plt.title('Percentage Error Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 8)
        
        pred_bins = np.percentile(self.predictions, [0, 25, 50, 75, 100])
        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        accuracies = []
        
        for i in range(len(pred_bins)-1):
            mask = (self.predictions >= pred_bins[i]) & (self.predictions < pred_bins[i+1])
            if i == len(pred_bins)-2:
                mask = (self.predictions >= pred_bins[i]) & (self.predictions <= pred_bins[i+1])
            
            if mask.sum() > 0:
                errors = np.abs(y_test_original[mask] - self.predictions[mask])
                relative_errors = errors / y_test_original[mask]
                accuracy = np.mean(relative_errors <= 0.1) * 100
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        plt.bar(bin_labels, accuracies, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
        plt.title('Accuracy by Prediction Quartile', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (% within 10%)')
        plt.xlabel('Prediction Quartile')
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 3, 9)
        if hasattr(self, 'feature_columns') and len(self.feature_columns) <= 20:
            
            first_layer_weights = self.model.layers[0].get_weights()[0]
            feature_importance = np.abs(first_layer_weights).mean(axis=1)
            
            
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [self.feature_columns[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.barh(range(len(top_indices)), top_importance)
            plt.yticks(range(len(top_indices)), top_features)
            plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('Average Absolute Weight')
        else:
            plt.text(0.5, 0.5, 'Feature Importance\nNot Available\n(Too many features)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        
        plt.savefig('_tax_fairness_evaluation.png', dpi=300, bbox_inches='tight')
        logger.info(" evaluation plots saved as '_tax_fairness_evaluation.png'")
    
    def detect_anomalies(self, threshold_percentile=95):
        """ anomaly detection with multiple methods"""
        if self.predictions is None:
            logger.error("Model not evaluated. Call evaluate_model() first.")
            return None
        
        
        if self.target_scaler is not None:
            y_test_original = self.target_scaler.inverse_transform(
                self.y_test_scaled.reshape(-1, 1)
            ).flatten()
        else:
            y_test_original = self.y_test_scaled
        
        
        absolute_errors = np.abs(y_test_original - self.predictions)
        relative_errors = absolute_errors / y_test_original
        percentage_errors = relative_errors * 100
        
        
        thresholds = {
            'percentage': np.percentile(percentage_errors, threshold_percentile),
            'absolute': np.percentile(absolute_errors, threshold_percentile),
            'z_score': 3.0
        }
        
        
        z_scores = np.abs(stats.zscore(absolute_errors))
        
        
        anomaly_methods = {
            'percentage': percentage_errors > thresholds['percentage'],
            'absolute': absolute_errors > thresholds['absolute'],
            'z_score': z_scores > thresholds['z_score']
        }
        
        
        combined_anomalies = np.any(list(anomaly_methods.values()), axis=0)
        
        logger.info(f"\n{'='*50}")
        logger.info(" ANOMALY DETECTION RESULTS")
        logger.info(f"{'='*50}")
        
        for method, anomalies in anomaly_methods.items():
            logger.info(f"{method.upper()} method:")
            logger.info(f"  Threshold: {thresholds.get(method, 'N/A')}")
            logger.info(f"  Anomalies detected: {np.sum(anomalies)}")
            logger.info(f"  Anomaly rate: {np.mean(anomalies)*100:.2f}%")
        
        logger.info(f"\nCOMBINED RESULTS:")
        logger.info(f"  Total anomalies: {np.sum(combined_anomalies)}")
        logger.info(f"  Overall anomaly rate: {np.mean(combined_anomalies)*100:.2f}%")
        
        return {
            'anomalies_combined': combined_anomalies,
            'anomalies_by_method': anomaly_methods,
            'anomaly_indices': np.where(combined_anomalies)[0],
            'percentage_errors': percentage_errors[combined_anomalies],
            'absolute_errors': absolute_errors[combined_anomalies],
            'thresholds': thresholds,
            'test_data': self.X_test.iloc[combined_anomalies] if hasattr(self.X_test, 'iloc') else self.X_test[combined_anomalies]
        }
    
    def save_model(self, model_path='_tax_fairness_model.keras'):
        """ model saving with comprehensive metadata"""
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return False
        
        
        self.model.save(model_path)
        
        
        joblib.dump(self.scaler, '_tax_model_scaler.pkl')
        if self.target_scaler is not None:
            joblib.dump(self.target_scaler, '_tax_target_scaler.pkl')
        
        
        metadata = {
            'model_version': '3.0_',
            'input_shape': self.X_train_scaled.shape[1],
            'feature_columns': self.feature_columns,
            'model_path': model_path,
            'scaler_path': '_tax_model_scaler.pkl',
            'target_scaler_path': '_tax_target_scaler.pkl' if self.target_scaler else None,
            'training_samples': len(self.X_train_scaled),
            'validation_samples': len(self.X_val_scaled),
            'test_samples': len(self.X_test_scaled),
            'model_architecture': {
                'layers': len(self.model.layers),
                'total_params': self.model.count_params(),
                'optimizer': 'Adam',
                'loss': 'Huber',
                'regularization': 'L1_L2'
            }
        }
        joblib.dump(metadata, '_tax_model_metadata.pkl')
        
        logger.info(f" model saved successfully:")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Feature Scaler: _tax_model_scaler.pkl")
        if self.target_scaler is not None:
            logger.info(f"  - Target Scaler: _tax_target_scaler.pkl")
        logger.info(f"  - Metadata: _tax_model_metadata.pkl")
        
        return True
    
    def run_complete_pipeline(self):
        """Run the complete  training and evaluation pipeline"""
        logger.info("Starting  Tax Fairness ANN Pipeline...")

        if os.path.exists('_tax_fairness_model.keras'):
            self.model = tf.keras.models.load_model('_tax_fairness_model.keras')
            logger.info("Loaded existing tax fairness model. Proceeding to evaluation and saving.")
        else:
            if not self.load_and_prepare_data(): return False
            if not self.train_model(): return False

        
        steps = [
            (" Data Loading & Preparation", self.load_and_prepare_data),
            (" Model Training", lambda: self.train_model(epochs=300, batch_size=64)),
            ("Comprehensive Model Evaluation", self.evaluate_model),
            (" Results Visualization", self.visualize_results),
            ("Advanced Anomaly Detection", self.detect_anomalies),
            (" Model Saving", self.save_model)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*70}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*70}")
            
            if step_name in [" Results Visualization", "Advanced Anomaly Detection"]:
                step_func()
            else:
                result = step_func()
                if result is False:
                    logger.error(f"Failed at step: {step_name}")
                    return False
        
        logger.info(f"\n{'='*70}")
        logger.info("  TAX FAIRNESS ANN PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*70}")
        logger.info("\n KEY IMPROVEMENTS IMPLEMENTED:")
        logger.info("   ✓ Advanced data preprocessing with multivariate outlier detection")
        logger.info("   ✓  model architecture with regularization and batch normalization")
        logger.info("   ✓ Robust scaling and target transformation")
        logger.info("   ✓ Three-way data split (train/validation/test)")
        logger.info("   ✓ Huber loss for outlier robustness")
        logger.info("   ✓ Comprehensive evaluation metrics")
        logger.info("   ✓  visualization suite")
        logger.info("   ✓ Multi-method anomaly detection")
        logger.info("   ✓ Safe percentage error calculations")

        self.evaluate_model()
        self.save_model()
        return True



def load_saved_model(model_path='_tax_fairness_model.keras'):
    """Load a saved  model with all components"""
    try:
        
        model = tf.keras.models.load_model(model_path)
        
        
        scaler = joblib.load('_tax_model_scaler.pkl')
        
        target_scaler = None
        if os.path.exists('_tax_target_scaler.pkl'):
            target_scaler = joblib.load('_tax_target_scaler.pkl')
        
        
        metadata = joblib.load('_tax_model_metadata.pkl')
        
        logger.info(" model loaded successfully!")
        logger.info(f"Model version: {metadata.get('model_version', 'Unknown')}")
        logger.info(f"Features: {len(metadata.get('feature_columns', []))}")
        
        return {
            'model': model,
            'scaler': scaler,
            'target_scaler': target_scaler,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def predict_tax_levy(model_components, property_data):
    """Make predictions using the loaded  model"""
    try:
        model = model_components['model']
        scaler = model_components['scaler']
        target_scaler = model_components['target_scaler']
        metadata = model_components['metadata']
        
        
        expected_features = metadata['feature_columns']
        
        
        property_scaled = scaler.transform(property_data)
        
        
        prediction_scaled = model.predict(property_scaled, verbose=0)
        
        
        if target_scaler is not None:
            prediction = target_scaler.inverse_transform(prediction_scaled).flatten()
        else:
            prediction = prediction_scaled.flatten()
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

def analyze_feature_importance(model_components, feature_names=None):
    """Analyze feature importance using model weights"""
    try:
        model = model_components['model']
        metadata = model_components['metadata']
        
        if feature_names is None:
            feature_names = metadata.get('feature_columns', [])
        
        
        first_layer = model.layers[0]
        weights = first_layer.get_weights()[0]
        
        
        importance = np.abs(weights).mean(axis=1)
        
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature Importance Analysis:")
        logger.info(importance_df.head(10))
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        return None


import os
from scipy import stats


if __name__ == "__main__":
    
    _tax_ann = TaxFairnessANN()
    success = _tax_ann.run_complete_pipeline()
    
    if success:
        print("\n  Tax Fairness ANN model completed successfully!")
        print(" Model saved and ready for deployment!")
        print("\n Key Improvements:")
        print("   • Better data preprocessing and outlier handling")
        print("   • Advanced model architecture with regularization")
        print("   • Robust training strategy with validation")
        print("   • Comprehensive evaluation and visualization")
        print("   •  anomaly detection capabilities")
        print("   • Safe error calculations (no more infinite MAPE!)")
        print("\n Your model is now production-ready!")
    else:
        print("\n  Tax Fairness ANN pipeline failed!")
        print("Please check the logs for detailed error information.")