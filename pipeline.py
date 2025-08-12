
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import os
import joblib
from datetime import datetime
import json
import tensorflow as tf


try:
    from dataset_preparation import PropertyDatasetPreprocessor
    from tax_ann_model import TaxFairnessANN
    from zoning_classifier_ann import ZoningClassifierANN
    from investment_risk_mlp import InvestmentRiskMLP

except ImportError as e:
    print(f" Warning: Custom modules not found: {e}")
    print("Make sure all files are in the same directory.")

warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('property_integrity_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PropertyInvestmentAnalyzer:
    def __init__(self, raw_data_path='property_tax_report.xlsx', force_retrain=False):
        self.raw_data_path = raw_data_path
        self.prepared_data_path = 'prepared_property_dataset.csv'
        self.force_retrain = force_retrain

        self.model_paths = {
            'tax_fairness': '_tax_fairness_model.keras',
            'zoning_classifier': 'zoning_classifier.keras',
            'investment_risk': 'investment_risk_mlp_model.keras'
        }

        self.preprocessor = None
        self.tax_model = None
        self.zoning_model = None
        self.investment_model = None

        self.results = {
            'preprocessing': None,
            'tax_fairness': None,
            'zoning_classification': None,
            'investment_risk': None
        }

        self.output_dir = f"property_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory created: {self.output_dir}")

    def check_existing_models(self):
        existing_models = {}
        for model_name, model_path in self.model_paths.items():
            exists = os.path.exists(model_path)
            existing_models[model_name] = exists
            logger.info(f"Model {model_name}: {' EXISTS' if exists else ' NOT FOUND'}")
        return existing_models

    def run_data_preprocessing(self):
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: DATA PREPROCESSING")
        logger.info("="*70)

        if os.path.exists(self.prepared_data_path) and not self.force_retrain:
            logger.info(" Skipping preprocessing: Prepared dataset already exists.")
            self.results['preprocessing'] = {
                'status': 'skipped',
                'output_file': self.prepared_data_path,
                'timestamp': datetime.now().isoformat()
            }
            return True

        try:
            logger.info(" Running data preprocessing...")
            self.preprocessor = PropertyDatasetPreprocessor(self.raw_data_path)
            success = self.preprocessor.run_preprocessing_pipeline()

            if success:
                self.results['preprocessing'] = {
                    'status': 'success',
                    'output_file': self.prepared_data_path,
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(" Data preprocessing completed successfully!")
                return True
            else:
                self.results['preprocessing'] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(" Data preprocessing failed!")
                return False

        except Exception as e:
            logger.error(f" Error in preprocessing: {str(e)}")
            self.results['preprocessing'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False

    def run_tax_fairness_analysis(self):
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: TAX FAIRNESS ANALYSIS")
        logger.info("="*70)

        try:
            if os.path.exists(self.model_paths['tax_fairness']) and not self.force_retrain:
                logger.info(" Tax fairness model already exists. Skipping training.")
                self.tax_model = TaxFairnessANN(self.prepared_data_path)
                self.tax_model.model = tf.keras.models.load_model(self.model_paths['tax_fairness'])
                self.results['tax_fairness'] = {
                    'status': 'loaded',
                    'model_path': self.model_paths['tax_fairness'],
                    'timestamp': datetime.now().isoformat()
                }
                return True

            self.tax_model = TaxFairnessANN(self.prepared_data_path)
            logger.info(" Running tax fairness analysis...")
            success = self.tax_model.run_complete_pipeline()

            if success:
                metrics = self.tax_model.evaluate_model()
                anomalies = self.tax_model.detect_anomalies()
                self.results['tax_fairness'] = {
                    'status': 'trained',
                    'metrics': metrics,
                    'anomalies_detected': len(anomalies['anomaly_indices']) if anomalies else 0,
                    'model_path': self.model_paths['tax_fairness'],
                    'timestamp': datetime.now().isoformat()
                }
                return True
            else:
                self.results['tax_fairness'] = {'status': 'failed', 'timestamp': datetime.now().isoformat()}
                return False

        except Exception as e:
            logger.error(f" Error in tax fairness analysis: {str(e)}")
            self.results['tax_fairness'] = {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
            return False

    def run_zoning_classification(self):
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: ZONING CLASSIFICATION ANALYSIS")
        logger.info("="*70)

        try:
            if os.path.exists(self.model_paths['zoning_classifier']) and not self.force_retrain:
                logger.info(" Zoning classifier model already exists. Skipping training.")
                self.zoning_model = ZoningClassifierANN(self.prepared_data_path, self.raw_data_path)
                self.zoning_model.model = tf.keras.models.load_model(self.model_paths['zoning_classifier'])
                self.results['zoning_classification'] = {
                    'status': 'loaded',
                    'model_path': self.model_paths['zoning_classifier'],
                    'timestamp': datetime.now().isoformat()
                }
                return True

            raw_csv_path = self.raw_data_path.replace('.xlsx', '.csv')
            raw_file = raw_csv_path if os.path.exists(raw_csv_path) else self.raw_data_path
            self.zoning_model = ZoningClassifierANN(self.prepared_data_path, raw_file)

            logger.info(" Running zoning classification analysis...")
            success = self.zoning_model.run_complete_pipeline()

            if success:
                metrics = self.zoning_model.evaluate_model()
                misclassifications = self.zoning_model.detect_misclassifications()
                self.results['zoning_classification'] = {
                    'status': 'trained',
                    'metrics': metrics,
                    'misclassifications_detected': len(misclassifications['misclassified_indices']) if misclassifications else 0,
                    'model_path': self.model_paths['zoning_classifier'],
                    'timestamp': datetime.now().isoformat()
                }
                return True
            else:
                self.results['zoning_classification'] = {'status': 'failed', 'timestamp': datetime.now().isoformat()}
                return False

        except Exception as e:
            logger.error(f" Error in zoning classification: {str(e)}")
            self.results['zoning_classification'] = {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
            return False

    def run_investment_risk_analysis(self):
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: INVESTMENT RISK ANALYSIS")
        logger.info("="*70)

        try:
            if os.path.exists(self.model_paths['investment_risk']) and not self.force_retrain:
                logger.info(" Investment risk model already exists. Skipping training.")
                self.investment_model = InvestmentRiskMLP(self.prepared_data_path)
                self.investment_model.model = tf.keras.models.load_model(self.model_paths['investment_risk'])
                self.results['investment_risk'] = {
                    'status': 'loaded',
                    'model_path': self.model_paths['investment_risk'],
                    'timestamp': datetime.now().isoformat()
                }
                return True

            self.investment_model = InvestmentRiskMLP(self.prepared_data_path)
            logger.info(" Running investment risk analysis...")
            success = self.investment_model.run_complete_pipeline()

            if success:
                metrics = self.investment_model.evaluate_model()
                risk_analysis = self.investment_model.analyze_risk_factors()
                recommendations = self.investment_model.generate_investment_recommendations()
                rec_counts = {}
                if isinstance(recommendations, list):
                    for rec in recommendations:
                        category = rec.get('risk_category', 'Unknown')
                        rec_counts[category] = rec_counts.get(category, 0) + 1

                self.results['investment_risk'] = {
                    'status': 'trained',
                    'metrics': metrics,
                    'risk_distribution': risk_analysis['risk_distribution'].to_dict() if risk_analysis else {},
                    'recommendation_counts': rec_counts,
                    'model_path': self.model_paths['investment_risk'],
                    'timestamp': datetime.now().isoformat()
                }
                return True
            else:
                self.results['investment_risk'] = {'status': 'failed', 'timestamp': datetime.now().isoformat()}
                return False

        except Exception as e:
            logger.error(f" Error in investment risk analysis: {str(e)}")
            self.results['investment_risk'] = {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
            return False
        

    def generate_comprehensive_report(self):
        """Print a summary report to the console."""
        logger.info("\n" + "="*70)
        logger.info("SUMMARY REPORT")
        logger.info("="*70)

        for phase, result in self.results.items():
            if result:
                status = result.get('status', 'unknown').upper()
                logger.info(f"{phase.replace('_', ' ').title()}: {status}")
            if 'metrics' in result:
                logger.info("  Metrics:")
                for key, value in result['metrics'].items():
                    logger.info(f"    - {key}: {value}")
            if 'anomalies_detected' in result:
                logger.info(f"  Anomalies Detected: {result['anomalies_detected']}")
            if 'misclassifications_detected' in result:
                logger.info(f"  Misclassifications: {result['misclassifications_detected']}")
            if 'recommendation_counts' in result:
                logger.info("  Investment Recommendations:")
                for category, count in result['recommendation_counts'].items():
                    logger.info(f"    - {category}: {count}")
        else:
            logger.info(f"{phase.replace('_', ' ').title()}: NO DATA")


if __name__ == "__main__":
    logger.info("Starting Property Integrity Suite (Auto Mode)")
    analyzer = PropertyInvestmentAnalyzer(force_retrain=False)

    if analyzer.run_data_preprocessing():
        analyzer.run_tax_fairness_analysis()
        analyzer.run_zoning_classification()
        analyzer.run_investment_risk_analysis()
        analyzer.generate_comprehensive_report()
        logger.info("Property Integrity Suite completed successfully!")
    else:
        logger.error("Failed to preprocess data. Aborting pipeline.")
