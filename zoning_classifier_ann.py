
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import joblib
import logging
import warnings
import re
import os
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZoningClassifierANN:
    def __init__(self, prepared_data_path='prepared_property_dataset.csv', raw_data_path='property_tax_report.csv'):
        """Initialize the Upgraded Zoning Classifier ANN"""
        self.prepared_data_path = prepared_data_path
        self.raw_data_path = raw_data_path
        self.model = None
        self.scaler = RobustScaler()  
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None
        self.predictions = None
        self.class_names = None
        self.baseline_score = None
        
    def load_and_prepare_data(self):
        """Load and prepare data with   preprocessing"""
        try:
            
            prepared_data = pd.read_csv(self.prepared_data_path)
            logger.info(f"Prepared dataset loaded. Shape: {prepared_data.shape}")
            
            
            zoning_data = self._load_zoning_data(prepared_data)
            if zoning_data is None:
                return False
            
            
            min_length = min(len(prepared_data), len(zoning_data))
            prepared_data = prepared_data.iloc[:min_length]
            zoning_data = zoning_data.iloc[:min_length]
            
            
            X = self._prepare_features(prepared_data)
            
            
            y_grouped = self._group_zoning_classes(zoning_data)
            
            
            self._data_quality_checks(X, y_grouped)
            
            
            X_filtered, y_filtered = self._filter_small_classes(X, y_grouped, min_samples=15)
            
            
            y_encoded = self.label_encoder.fit_transform(y_filtered)
            self.class_names = self.label_encoder.classes_
            
            logger.info(f"Final class distribution:")
            unique, counts = np.unique(y_encoded, return_counts=True)
            class_dist = dict(zip(self.class_names[unique], counts))
            for cls, count in class_dist.items():
                logger.info(f"  {cls}: {count} samples")
            
            
            X_selected = self._select_best_features(X_filtered, y_encoded)
            
            
            self._train_baseline_model(X_selected, y_encoded)
            
            
            X_balanced, y_balanced = self._balance_classes(X_selected, y_encoded)
            
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, 
                stratify=y_balanced if len(np.unique(y_balanced)) > 1 else None
            )
            
            
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            logger.info(f"Final training set shape: {self.X_train_scaled.shape}")
            logger.info(f"Final test set shape: {self.X_test_scaled.shape}")
            logger.info(f"Number of classes: {len(self.class_names)}")
            logger.info(f"Number of features after selection: {self.X_train_scaled.shape[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_zoning_data(self, prepared_data):
        """Load zoning data from raw or prepared sources"""
        try:
            
            try:
                raw_data = pd.read_csv(self.raw_data_path)
            except:
                raw_data = pd.read_csv(self.raw_data_path, sep=';')
            
            
            zoning_cols = [col for col in raw_data.columns if 'ZONING' in col.upper()]
            if zoning_cols:
                zoning_col = zoning_cols[0]
                zoning_data = raw_data[zoning_col].copy()
                logger.info(f"Found zoning column: {zoning_col}")
                return zoning_data
            
        except Exception as e:
            logger.warning(f"Could not load raw data: {e}")
        
        
        zoning_cols = [col for col in prepared_data.columns if 'ZONING' in col.upper()]
        if zoning_cols:
            return self._reconstruct_zoning_from_onehot(prepared_data, zoning_cols)
        
        logger.error("No zoning information found")
        return None
    
    def _prepare_features(self, prepared_data):
        """  feature preparation"""
        
        cols_to_remove = ['TAX_LEVY'] + [col for col in prepared_data.columns if 'ZONING' in col.upper()]
        X = prepared_data.drop(columns=cols_to_remove, errors='ignore')
        
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col].astype(str))[0]
        
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        
        if len(X.columns) > 1:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > 0.95)]
            if high_corr_features:
                logger.info(f"Removing {len(high_corr_features)} highly correlated features")
                X = X.drop(columns=high_corr_features)
        
        return X
    
    def _group_zoning_classes(self, zoning_data):
        """Group similar zoning classes to reduce complexity"""
        logger.info("Grouping zoning classes...")
        
        y = zoning_data.fillna('Unknown').astype(str).str.upper().str.strip()
        
        
        zoning_groups = {
            'RESIDENTIAL_LOW': ['R1', 'R-1', 'RR', 'RS', 'R1A', 'R1B', 'SFR', 'SINGLE'],
            'RESIDENTIAL_MED': ['R2', 'R-2', 'R3', 'R-3', 'RM', 'RD', 'DUPLEX', 'MULTI'],
            'RESIDENTIAL_HIGH': ['R4', 'R-4', 'R5', 'R-5', 'RH', 'APARTMENT', 'CONDO'],
            'COMMERCIAL': ['C1', 'C-1', 'C2', 'C-2', 'C3', 'C-3', 'COMM', 'COMMERCIAL', 'RETAIL'],
            'INDUSTRIAL': ['I1', 'I-1', 'I2', 'I-2', 'IND', 'INDUSTRIAL', 'MANUFACTURING'],
            'MIXED_USE': ['MU', 'MX', 'MIXED', 'PUD', 'PLANNED'],
            'AGRICULTURAL': ['A1', 'A-1', 'AG', 'AGRICULTURE', 'FARM', 'RURAL'],
            'PUBLIC': ['PUB', 'PUBLIC', 'SCHOOL', 'PARK', 'GOVERNMENT'],
            'OTHER': []
        }
        
        y_grouped = pd.Series(['OTHER'] * len(y), index=y.index)
        
        for group_name, keywords in zoning_groups.items():
            if group_name != 'OTHER':
                pattern = '|'.join([re.escape(kw) for kw in keywords])
                mask = y.str.contains(pattern, na=False, regex=True)
                y_grouped[mask] = group_name
        
        
        original_unique = y.nunique()
        grouped_unique = y_grouped.nunique()
        logger.info(f"Reduced classes from {original_unique} to {grouped_unique}")
        
        group_dist = Counter(y_grouped)
        for group, count in sorted(group_dist.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {group}: {count} samples")
        
        return y_grouped
    
    def _data_quality_checks(self, X, y):
        """Perform data quality checks"""
        logger.info("\nData Quality Checks:")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Missing values in features: {X.isnull().sum().sum()}")
        logger.info(f"Missing values in target: {y.isnull().sum()}")
        logger.info(f"Unique classes: {y.nunique()}")
        
        
        if len(X.columns) > 0 and len(y) > 0:
            potential_leakage = []
            for col in X.select_dtypes(include=[np.number]).columns[:10]:  
                try:
                    correlation = abs(pd.Series(y.factorize()[0]).corr(X[col].fillna(0)))
                    if correlation > 0.95:
                        potential_leakage.append((col, correlation))
                except:
                    pass
            
            if potential_leakage:
                logger.warning(f"Potential data leakage detected: {potential_leakage}")
    
    def _filter_small_classes(self, X, y, min_samples=15):
        """Filter out classes with too few samples"""
        logger.info(f"Filtering classes with fewer than {min_samples} samples...")
        
        class_counts = Counter(y)
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
        removed_classes = [cls for cls, count in class_counts.items() if count < min_samples]
        
        if removed_classes:
            removed_count = sum(class_counts[cls] for cls in removed_classes)
            logger.info(f"Removing {len(removed_classes)} classes with {removed_count} total samples")
            logger.info(f"Removed classes: {removed_classes}")
        
        mask = y.isin(valid_classes)
        X_filtered = X[mask].reset_index(drop=True)
        y_filtered = y[mask].reset_index(drop=True)
        
        logger.info(f"After filtering: {len(X_filtered)} samples, {len(valid_classes)} classes")
        return X_filtered, y_filtered
    
    def _select_best_features(self, X, y, max_features=50):
        """Select best features using statistical tests"""
        if X.shape[1] <= max_features:
            logger.info(f"No feature selection needed ({X.shape[1]} <= {max_features})")
            return X
        
        logger.info(f"Selecting top {max_features} features from {X.shape[1]}...")
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(max_features, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        logger.info(f"Selected features: {selected_features[:10]}...")  # Show first 10
        
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def _train_baseline_model(self, X, y):
        """Train baseline Random Forest for comparison"""
        logger.info("Training baseline Random Forest model...")
        
        try:
            X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X_train_base, y_train_base)
            
            self.baseline_score = rf.score(X_test_base, y_test_base)
            logger.info(f"Baseline Random Forest accuracy: {self.baseline_score:.3f}")
            
            if self.baseline_score < 0.3:
                logger.warning(" Very low baseline score - data quality issues likely")
            elif self.baseline_score < 0.5:
                logger.warning(" Low baseline score - challenging classification problem")
            else:
                logger.info(" Reasonable baseline score - ANN should perform well")
                
        except Exception as e:
            logger.warning(f"Baseline model training failed: {e}")
            self.baseline_score = None
    
    def _reconstruct_zoning_from_onehot(self, data, zoning_cols):
        """Reconstruct original zoning categories from one-hot encoded columns"""
        zoning_data = []
        for idx in range(len(data)):
            found_class = "Unknown"
            for col in zoning_cols:
                if data[col].iloc[idx] == 1:
                    found_class = col.replace('ZONING_CLASSIFICATION_', '').replace('ZONING_', '')
                    break
            zoning_data.append(found_class)
        return pd.Series(zoning_data)
    
    def _balance_classes(self, X, y):
        """  class balancing with SMOTE"""
        logger.info("\nBalancing classes...")
        
        original_dist = Counter(y)
        logger.info(f"Original distribution: {dict(original_dist)}")
        
        min_class_size = min(original_dist.values())
        if min_class_size < 2:
            logger.warning("Classes too small for SMOTE, using original data")
            return X, y
        
        try:
            
            k_neighbors = min(3, min_class_size - 1)
            logger.info(f"Using k_neighbors={k_neighbors} for SMOTE")
            
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            balanced_dist = Counter(y_balanced)
            logger.info(f"Balanced distribution: {dict(balanced_dist)}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y
    
    def build_model(self, input_dim, num_classes):
        """Build optimized model architecture"""
        tf.keras.backend.clear_session()
        
        
        if num_classes <= 3:
            
            model = Sequential([
                Dense(32, input_dim=input_dim, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dropout(0.1),
                Dense(num_classes, activation='softmax')
            ])
            lr = 0.01
        elif num_classes <= 6:
            
            model = Sequential([
                Dense(64, input_dim=input_dim, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dropout(0.1),
                Dense(num_classes, activation='softmax')
            ])
            lr = 0.005
        else:
            
            model = Sequential([
                Dense(128, input_dim=input_dim, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(num_classes, activation='softmax')
            ])
            lr = 0.001
        
        
        optimizer = Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built model with {num_classes} classes, learning rate: {lr}")
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with   callbacks"""
        if self.X_train_scaled is None:
            logger.error("Data not prepared. Call load_and_prepare_data() first.")
            return False
        
        num_classes = len(self.class_names)
        self.model = self.build_model(self.X_train_scaled.shape[1], num_classes)
        
        
        logger.info(f"\nModel Architecture Summary:")
        logger.info(f"Input features: {self.X_train_scaled.shape[1]}")
        logger.info(f"Output classes: {num_classes}")
        logger.info(f"Total parameters: {self.model.count_params()}")
        
        
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(self.y_train), y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_zoning_classifier.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        
        if len(self.X_train_scaled) < 1000:
            batch_size = 16
        elif len(self.X_train_scaled) > 10000:
            batch_size = 64
        
        logger.info(f"Training with batch_size={batch_size}, epochs={epochs}")
        
        
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed!")
        return True
    
    def evaluate_model(self):
        """  model evaluation with baseline comparison"""
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return False
        
        
        y_pred_probs = self.model.predict(self.X_test_scaled, verbose=0)
        self.predictions = np.argmax(y_pred_probs, axis=1)
        
        
        accuracy = accuracy_score(self.y_test, self.predictions)
        f1_weighted = f1_score(self.y_test, self.predictions, average='weighted')
        f1_macro = f1_score(self.y_test, self.predictions, average='macro')
        
        
        logger.info("\n" + "="*60)
        logger.info("  CLASSIFICATION EVALUATION")
        logger.info("="*60)
        
        if self.baseline_score:
            logger.info(f"Baseline (Random Forest): {self.baseline_score:.4f}")
            improvement = accuracy - self.baseline_score
            logger.info(f"ANN Improvement: {improvement:+.4f}")
        
        logger.info(f"ANN Accuracy: {accuracy:.4f}")
        logger.info(f"F1-Score (Weighted): {f1_weighted:.4f}")
        logger.info(f"F1-Score (Macro): {f1_macro:.4f}")
        
        
        if accuracy > 0.8:
            logger.info("🎉 Excellent performance!")
        elif accuracy > 0.6:
            logger.info(" Good performance")
        elif accuracy > 0.4:
            logger.info(" Moderate performance")
        else:
            logger.info(" Poor performance - consider data quality issues")
        
        
        logger.info("\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.predictions, target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'baseline_score': self.baseline_score,
            'improvement': accuracy - self.baseline_score if self.baseline_score else None
        }
    
    def visualize_results(self):
        """  visualization with more insights"""
        if self.history is None or self.predictions is None:
            logger.error("Model not trained or evaluated.")
            return
        
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 18))
        
        
        plt.subplot(3, 4, 1)
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        plt.plot(epochs, self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy History', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        plt.plot(epochs, self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Model Loss History', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 4, 3)
        cm = confusion_matrix(self.y_test, self.predictions)
        sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=self.class_names, 
                yticklabels=self.class_names, 
                cmap='Blues', square=True)
        plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        
        plt.subplot(3, 4, 4)
        report = classification_report(self.y_test, self.predictions, 
                                    target_names=self.class_names, output_dict=True)
        
        classes_in_report = [name for name in self.class_names if name in report]
        f1_scores = [report[class_name]['f1-score'] for class_name in classes_in_report]
        
        bars = plt.bar(range(len(f1_scores)), f1_scores, 
                    color='orange', edgecolor='black', alpha=0.7)
        plt.title('Per-Class F1-Score', fontsize=12, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.xticks(range(len(f1_scores)), classes_in_report, rotation=45)
        plt.ylim(0, 1)
        
        for bar, value in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        
        plt.subplot(3, 4, 5)
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        pred_dist = pd.Series(self.predictions).value_counts().sort_index()
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        actual_counts = [test_dist.get(i, 0) for i in range(len(self.class_names))]
        pred_counts = [pred_dist.get(i, 0) for i in range(len(self.class_names))]
        
        plt.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.7, color='blue')
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='red')
        
        plt.title('Actual vs Predicted Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(x, self.class_names, rotation=45)
        plt.legend()
        
        
        plt.subplot(3, 4, 6)
        y_pred_probs = self.model.predict(self.X_test_scaled, verbose=0)
        confidence_scores = np.max(y_pred_probs, axis=1)
        
        plt.hist(confidence_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
        plt.title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidence_scores):.3f}')
        plt.legend()
        
        
        plt.subplot(3, 4, 7)
        if 'lr' in self.history.history:
            plt.plot(epochs, self.history.history['lr'], 'g-', linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nHistory\nNot Available', 
                    ha='center', va='center', fontsize=12)
            plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        
        
        plt.subplot(3, 4, 8)
        metrics = self.evaluate_model()
        
        metric_names = ['Accuracy', 'F1-Weighted', 'F1-Macro']
        metric_values = [metrics['accuracy'], metrics['f1_weighted'], metrics['f1_macro']]
        
        if self.baseline_score:
            metric_names.append('Baseline')
            metric_values.append(self.baseline_score)
        
        bars = plt.bar(metric_names, metric_values, 
                    color=['blue', 'green', 'orange', 'red'][:len(metric_names)],
                    alpha=0.7, edgecolor='black')
        
        plt.title('Model Performance Summary', fontsize=12, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        
        plt.subplot(3, 4, 9)
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            try:
                
                feature_scores = self.feature_selector.scores_
                selected_features = self.feature_selector.get_support()
                
                if len(feature_scores) > 0:
                    
                    top_indices = np.argsort(feature_scores[selected_features])[-10:]
                    top_scores = feature_scores[selected_features][top_indices]
                    
                    
                    feature_names = [f'Feature_{i}' for i in top_indices]
                    
                    plt.barh(range(len(top_scores)), top_scores, color='purple', alpha=0.7)
                    plt.yticks(range(len(top_scores)), feature_names)
                    plt.title('Top 10 Feature Scores', fontsize=12, fontweight='bold')
                    plt.xlabel('Score')
                else:
                    plt.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                            ha='center', va='center', fontsize=12)
            except:
                plt.text(0.5, 0.5, 'Feature Importance\nCalculation Failed', 
                        ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, 'Feature Selection\nNot Performed', 
                    ha='center', va='center', fontsize=12)
        plt.title('Feature Importance', fontsize=12, fontweight='bold')
        
        
        plt.subplot(3, 4, 10)
        misclassified = self.y_test != self.predictions
        error_by_class = {}
        
        for i, class_name in enumerate(self.class_names):
            class_mask = self.y_test == i
            if np.sum(class_mask) > 0:
                error_rate = np.sum(misclassified & class_mask) / np.sum(class_mask)
                error_by_class[class_name] = error_rate
        
        if error_by_class:
            classes = list(error_by_class.keys())
            error_rates = list(error_by_class.values())
            
            bars = plt.bar(range(len(classes)), error_rates, 
                        color='red', alpha=0.7, edgecolor='black')
            plt.title('Error Rate by Class', fontsize=12, fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('Error Rate')
            plt.xticks(range(len(classes)), classes, rotation=45)
            plt.ylim(0, 1)
            
            for bar, value in zip(bars, error_rates):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        
        plt.subplot(3, 4, 11)
        if len(self.history.history['accuracy']) > 5:
            
            early_acc = np.mean(self.history.history['val_accuracy'][:5])
            late_acc = np.mean(self.history.history['val_accuracy'][-5:])
            improvement = late_acc - early_acc
            
            
            smoothed_val_acc = pd.Series(self.history.history['val_accuracy']).rolling(5).mean()
            plt.plot(epochs, smoothed_val_acc, 'b-', linewidth=2, label='Smoothed Val Acc')
            plt.axhline(y=early_acc, color='red', linestyle='--', alpha=0.7, label=f'Early: {early_acc:.3f}')
            plt.axhline(y=late_acc, color='green', linestyle='--', alpha=0.7, label=f'Final: {late_acc:.3f}')
            
            plt.title(f'Training Convergence\n(Improvement: {improvement:+.3f})',
                fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        
        plt.subplot(3, 4, 12)
        plt.axis('off')
        
        
        summary_text = "MODEL SUMMARY\n" + "="*15 + "\n"
        summary_text += f"Classes: {len(self.class_names)}\n"
        summary_text += f"Features: {self.X_train_scaled.shape[1]}\n"
        summary_text += f"Training Samples: {len(self.X_train_scaled)}\n"
        summary_text += f"Test Samples: {len(self.X_test_scaled)}\n\n"
        
        summary_text += "PERFORMANCE\n" + "-"*11 + "\n"
        summary_text += f"Accuracy: {metrics['accuracy']:.3f}\n"
        summary_text += f"F1-Weighted: {metrics['f1_weighted']:.3f}\n"
        summary_text += f"F1-Macro: {metrics['f1_macro']:.3f}\n"
        
        if self.baseline_score:
            summary_text += f"Baseline: {self.baseline_score:.3f}\n"
            improvement = metrics['accuracy'] - self.baseline_score
            summary_text += f"Improvement: {improvement:+.3f}\n"
        
        
        if metrics['accuracy'] > 0.8:
            summary_text += "\nSTATUS:  EXCELLENT"
        elif metrics['accuracy'] > 0.6:
            summary_text += "\nSTATUS:  GOOD"
        elif metrics['accuracy'] > 0.4:
            summary_text += "\nSTATUS:  MODERATE"
        else:
            summary_text += "\nSTATUS:  NEEDS WORK"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        
        plt.savefig('zoning_classifier_evaluation.png', dpi=300, bbox_inches='tight')
        logger.info("evaluation plots saved as 'zoning_classifier_evaluation.png'")
    
    def detect_misclassifications(self):
        """  misclassification analysis"""
        if self.predictions is None:
            logger.error("Model not evaluated. Call evaluate_model() first.")
            return None
        
        
        y_pred_probs = self.model.predict(self.X_test_scaled, verbose=0)
        confidence_scores = np.max(y_pred_probs, axis=1)
        
        
        misclassified_mask = self.y_test != self.predictions
        
        
        low_confidence_threshold = 0.5
        medium_confidence_threshold = 0.8
        
        low_conf_mask = confidence_scores < low_confidence_threshold
        medium_conf_mask = (confidence_scores >= low_confidence_threshold) & (confidence_scores < medium_confidence_threshold)
        high_conf_mask = confidence_scores >= medium_confidence_threshold
        
        logger.info(f"\n MISCLASSIFICATION ANALYSIS:")
        logger.info(f"{'='*50}")
        logger.info(f"Total predictions: {len(self.predictions)}")
        logger.info(f"Correct predictions: {np.sum(~misclassified_mask)} ({np.mean(~misclassified_mask)*100:.1f}%)")
        logger.info(f"Misclassifications: {np.sum(misclassified_mask)} ({np.mean(misclassified_mask)*100:.1f}%)")
        
        logger.info(f"\nConfidence Distribution:")
        logger.info(f"High confidence (≥{medium_confidence_threshold:.1f}): {np.sum(high_conf_mask)} ({np.mean(high_conf_mask)*100:.1f}%)")
        logger.info(f"Medium confidence ({low_confidence_threshold:.1f}-{medium_confidence_threshold:.1f}): {np.sum(medium_conf_mask)} ({np.mean(medium_conf_mask)*100:.1f}%)")
        logger.info(f"Low confidence (<{low_confidence_threshold:.1f}): {np.sum(low_conf_mask)} ({np.mean(low_conf_mask)*100:.1f}%)")
        
        
        high_conf_accuracy = np.mean(~misclassified_mask[high_conf_mask]) if np.sum(high_conf_mask) > 0 else 0
        medium_conf_accuracy = np.mean(~misclassified_mask[medium_conf_mask]) if np.sum(medium_conf_mask) > 0 else 0
        low_conf_accuracy = np.mean(~misclassified_mask[low_conf_mask]) if np.sum(low_conf_mask) > 0 else 0
        
        logger.info(f"\nAccuracy by Confidence Level:")
        logger.info(f"High confidence accuracy: {high_conf_accuracy:.3f}")
        logger.info(f"Medium confidence accuracy: {medium_conf_accuracy:.3f}")
        logger.info(f"Low confidence accuracy: {low_conf_accuracy:.3f}")
        
        
        logger.info(f"\nPer-Class Error Analysis:")
        for i, class_name in enumerate(self.class_names):
            class_mask = self.y_test == i
            if np.sum(class_mask) > 0:
                class_errors = np.sum(misclassified_mask & class_mask)
                class_total = np.sum(class_mask)
                error_rate = class_errors / class_total
                avg_confidence = np.mean(confidence_scores[class_mask])
                logger.info(f"  {class_name}: {error_rate:.3f} error rate, {avg_confidence:.3f} avg confidence ({class_total} samples)")
        
        return {
            'misclassified_indices': np.where(misclassified_mask)[0],
            'low_confidence_indices': np.where(low_conf_mask)[0],
            'confidence_scores': confidence_scores,
            'misclassification_rate': np.mean(misclassified_mask),
            'high_conf_accuracy': high_conf_accuracy,
            'medium_conf_accuracy': medium_conf_accuracy,
            'low_conf_accuracy': low_conf_accuracy
        }
    
    def save_model(self, model_path='zoning_classifier.keras'):
        """Save the   model and all components"""
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return False
        
        
        self.model.save(model_path)
        
        
        joblib.dump(self.scaler, 'zoning_scaler.pkl')
        joblib.dump(self.label_encoder, 'zoning_label_encoder.pkl')
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, 'zoning_feature_selector.pkl')
        
        
        metadata = {
            'model_version': '2.0_enhanced',
            'input_shape': self.X_train_scaled.shape[1],
            'num_classes': len(self.class_names),
            'class_names': self.class_names.tolist(),
            'baseline_score': self.baseline_score,
            'final_accuracy': self.evaluate_model()['accuracy'] if hasattr(self, 'predictions') and self.predictions is not None else None,
            'feature_selection_used': self.feature_selector is not None,
            'training_samples': len(self.X_train_scaled),
            'test_samples': len(self.X_test_scaled),
            'model_path': model_path,
            'scaler_path': 'zoning_scaler.pkl',
            'label_encoder_path': 'zoning_label_encoder.pkl',
            'feature_selector_path': 'zoning_feature_selector.pkl' if self.feature_selector else None
        }
        joblib.dump(metadata, 'zoning_metadata.pkl')
        
        logger.info(f"model saved successfully:")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Scaler: zoning_scaler.pkl")
        logger.info(f"  - Label Encoder: zoning_label_encoder.pkl")
        if self.feature_selector:
            logger.info(f"  - Feature Selector: zoning_feature_selector.pkl")
        logger.info(f"  - Metadata: zoning_metadata.pkl")
        
        return True
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        logger.info("Starting Zoning Classifier ANN Pipeline...")
        logger.info(" Version 2.0 - Optimized for Better Performance")

        skip_training = False

        if os.path.exists('zoning_classifier.keras'):
            self.model = tf.keras.models.load_model('zoning_classifier.keras')
            logger.info("Loaded existing zoning model.")
            skip_training = True

        steps = [
            (" Data Loading & Preparation", self.load_and_prepare_data),
            (" Model Training with Adaptive Architecture", lambda: self.train_model(epochs=100, batch_size=32) if not skip_training else True),
            (" Comprehensive Model Evaluation", self.evaluate_model),
            (" Results Visualization", self.visualize_results),
            (" Advanced Misclassification Analysis", self.detect_misclassifications),
            (" Model Saving", self.save_model)
        ]

        for step_name, step_func in steps:
            logger.info(f"\n{'='*70}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*70}")

            try:
                if step_name in [" Results Visualization", " Advanced Misclassification Analysis"]:
                    step_func()
                else:
                    result = step_func()
                    if result is False:
                        logger.error(f" Failed at step: {step_name}")
                        return False
                    else:
                        logger.info(f" Completed: {step_name}")
            except Exception as e:
                logger.error(f" Error in step '{step_name}': {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False

        logger.info(f"\n{'='*70}")
        logger.info(" ZONING CLASSIFIER ANN PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(" Check the evaluation plots for detailed analysis")
        logger.info(" Model saved with comprehensive metadata")
        logger.info(f"{'='*70}")

        return True

def load_enhanced_model(model_path='zoning_classifier.keras'):
    """Load the saved model for inference"""
    try:
        
        model = tf.keras.models.load_model(model_path)
        
        
        scaler = joblib.load('zoning_scaler.pkl')
        label_encoder = joblib.load('zoning_label_encoder.pkl')
        
        
        feature_selector = None
        try:
            feature_selector = joblib.load('zoning_feature_selector.pkl')
        except:
            pass
        
        
        metadata = joblib.load('enhanced_zoning_metadata.pkl')
        
        logger.info(" model loaded successfully!")
        logger.info(f"Model version: {metadata.get('model_version', 'Unknown')}")
        logger.info(f"Classes: {len(metadata['class_names'])}")
        logger.info(f"Features: {metadata['input_shape']}")
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_selector': feature_selector,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to load   model: {e}")
        return None


if __name__ == "__main__":
    classifier = ZoningClassifierANN()
    classifier.load_and_prepare_data()
    success = classifier.train_model()
    if success:
        classifier.evaluate_model()
        classifier.save_model()
    else:
        print(" Training failed")
