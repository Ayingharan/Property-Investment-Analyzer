

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyDatasetPreprocessor:
    def __init__(self, file_path):
        """Initialize the preprocessor with file path"""
        self.file_path = file_path
        self.dataset = None
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load dataset with proper error handling"""
        try:
            
            if self.file_path.endswith('.xlsx'):
                self.dataset = pd.read_excel(self.file_path, engine='openpyxl')
            elif self.file_path.endswith('.csv'):
                
                try:
                    self.dataset = pd.read_csv(self.file_path)
                except:
                    self.dataset = pd.read_csv(self.file_path, sep=';')
            
            logger.info(f"Dataset loaded successfully. Shape: {self.dataset.shape}")
            logger.info(f"Columns: {list(self.dataset.columns)}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def explore_data(self):
        """Explore dataset structure and quality"""
        if self.dataset is None:
            logger.error("Dataset not loaded")
            return
            
        logger.info("\n=== DATA EXPLORATION ===")
        logger.info(f"Dataset shape: {self.dataset.shape}")
        logger.info(f"Memory usage: {self.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        
        missing_info = self.dataset.isnull().sum()
        if missing_info.sum() > 0:
            logger.info(f"\nMissing values found:")
            for col, missing_count in missing_info[missing_info > 0].items():
                percentage = (missing_count / len(self.dataset)) * 100
                logger.info(f"  {col}: {missing_count} ({percentage:.2f}%)")
        
        
        logger.info(f"\nData types:")
        for col, dtype in self.dataset.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        
        logger.info(f"\nFirst 5 rows:")
        print(self.dataset.head())
        
        
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info(f"\nStatistical summary for numeric columns:")
            print(self.dataset[numeric_cols].describe())
    
    def clean_data(self):
        """Clean and handle missing values intelligently"""
        if self.dataset is None:
            logger.error("Dataset not loaded")
            return False
            
        logger.info("\n=== DATA CLEANING ===")
        initial_shape = self.dataset.shape
        
        
        for col in self.dataset.columns:
            if self.dataset[col].isnull().sum() > 0:
                if self.dataset[col].dtype == 'object':
                    
                    mode_value = self.dataset[col].mode()
                    if len(mode_value) > 0:
                        self.dataset[col].fillna(mode_value[0], inplace=True)
                    else:
                        self.dataset[col].fillna('Unknown', inplace=True)
                else:
                    
                    self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
        
        
        duplicates = self.dataset.duplicated().sum()
        if duplicates > 0:
            self.dataset.drop_duplicates(inplace=True)
            logger.info(f"Removed {duplicates} duplicate rows")
        
        
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.dataset[col].quantile(0.25)
            Q3 = self.dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.dataset[col] < lower_bound) | (self.dataset[col] > upper_bound)).sum()
            if outliers > 0:
                
                self.dataset[col] = np.clip(self.dataset[col], lower_bound, upper_bound)
                logger.info(f"Capped {outliers} outliers in {col}")
        
        logger.info(f"Dataset shape after cleaning: {self.dataset.shape} (removed {initial_shape[0] - self.dataset.shape[0]} rows)")
        return True
    
    def encode_categorical_features(self):
        """Encode categorical features with proper handling"""
        if self.dataset is None:
            logger.error("Dataset not loaded")
            return False
            
        logger.info("\n=== CATEGORICAL ENCODING ===")
        
        
        categorical_cols = self.dataset.select_dtypes(include=['object']).columns
        
        
        high_cardinality_threshold = 10
        
        for col in categorical_cols:
            unique_values = self.dataset[col].nunique()
            
            if unique_values <= high_cardinality_threshold:
                
                dummies = pd.get_dummies(self.dataset[col], prefix=col, drop_first=True)
                self.dataset = pd.concat([self.dataset, dummies], axis=1)
                logger.info(f"One-hot encoded {col} ({unique_values} unique values)")
            else:
                
                le = LabelEncoder()
                self.dataset[col + '_encoded'] = le.fit_transform(self.dataset[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Label encoded {col} ({unique_values} unique values)")
            
            
            self.dataset.drop(col, axis=1, inplace=True)
        
        return True
    
    def normalize_features(self):
        """Normalize numerical features"""
        if self.dataset is None:
            logger.error("Dataset not loaded")
            return False
            
        logger.info("\n=== FEATURE NORMALIZATION ===")
        
        
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        
        
        target_col = 'TAX_LEVY'
        if target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        if len(numeric_cols) > 0:
            
            self.dataset[numeric_cols] = self.scaler.fit_transform(self.dataset[numeric_cols])
            logger.info(f"Normalized {len(numeric_cols)} numeric features")
            
            
            for col in numeric_cols:
                logger.info(f"  {col}: min={self.dataset[col].min():.3f}, max={self.dataset[col].max():.3f}")
        
        return True
    
    def prepare_final_dataset(self):
        """Prepare final dataset for model training"""
        if self.dataset is None:
            logger.error("Dataset not loaded")
            return False
            
        logger.info("\n=== FINAL DATASET PREPARATION ===")
        
        
        if 'TAX_LEVY' not in self.dataset.columns:
            logger.error("Target variable 'TAX_LEVY' not found in dataset")
            return False
        
        
        X = self.dataset.drop(columns=['TAX_LEVY'])
        y = self.dataset['TAX_LEVY']
        
        
        final_dataset = pd.concat([X, y], axis=1)
        
        
        output_file = 'prepared_property_dataset.csv'
        final_dataset.to_csv(output_file, index=False)
        
        logger.info(f"Final dataset shape: {final_dataset.shape}")
        logger.info(f"Features: {len(X.columns)}")
        logger.info(f"Target variable: {y.name}")
        logger.info(f"Saved to: {output_file}")
        
        
        import joblib
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        
        return True
    
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        steps = [
            ("Loading data", self.load_data),
            ("Exploring data", self.explore_data),
            ("Cleaning data", self.clean_data),
            ("Encoding categorical features", self.encode_categorical_features),
            ("Normalizing features", self.normalize_features),
            ("Preparing final dataset", self.prepare_final_dataset)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*50}")
            
            if step_name == "Exploring data":
                step_func()
            else:
                if not step_func():
                    logger.error(f"Failed at step: {step_name}")
                    return False
        
        logger.info(f"\n{'='*50}")
        logger.info(" PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*50}")
        
        return True


if __name__ == "__main__":
    
    preprocessor = PropertyDatasetPreprocessor("property_tax_report.xlsx")
    
    
    success = preprocessor.run_preprocessing_pipeline()
    
    if success:
        print("\n Dataset preparation completed successfully!")
        print(" Cleaned file saved as: prepared_property_dataset.csv")
        print(" Preprocessing components saved for model deployment")
    else:
        print("\n Dataset preparation failed!")