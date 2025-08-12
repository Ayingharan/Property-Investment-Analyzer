

from smart_model_test import SmartPropertyModelLoader, PropertyPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyInvestmentAnalyzerPipeline:
    """Deployment pipeline using your trained models"""

    def __init__(self):
        self.model_loader = SmartPropertyModelLoader()
        self.predictor = None
        self.data = None
        self.results = None

    def initialize(self):
        """Initialize the pipeline with your models"""
        logger.info("Initializing Property Investment Analyzer Pipeline")
        logger.info("Using your trained ANN/MLP models...")

        if self.model_loader.load_all_models():
            self.predictor = PropertyPredictor(self.model_loader)
            logger.info("Pipeline initialized successfully!")
            return True
        else:
            logger.error("Failed to load one or more models!")
            return False

    def load_property_data(self, file_path=None):
        """Load property data for analysis"""

        
        if not file_path:
            data_files = [
                'prepared_property_dataset.csv',
                'property_tax_report.csv',
                'property_tax_report.xlsx'
            ]

            for file in data_files:
                if os.path.exists(file):
                    file_path = file
                    break

        if not file_path:
            logger.error("No property data file found!")
            return False

        try:
            logger.info(f"Loading data from: {file_path}")

            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)

            logger.info(f"Loaded {len(self.data)} properties")
            logger.info(f"Features: {len(self.data.columns)} columns")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def run_analysis(self):
        """Run complete property analysis"""
        if self.data is None or self.predictor is None:
            logger.error("Pipeline not ready. Call initialize() and load_property_data() first.")
            return False

        logger.info("Running property analysis...")
        logger.info(f"Analyzing {len(self.data)} properties with {len(self.model_loader.models)} models")

        
        self.results = self.predictor.predict_all(self.data)

        
        self._log_results_summary()

        return True

    def _log_results_summary(self):
        """Log analysis results summary"""
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS RESULTS SUMMARY")
        logger.info("="*60)

        if 'tax_fairness' in self.results:
            tax_results = self.results['tax_fairness']
            if 'error' not in tax_results:
                logger.info(f"Tax Analysis: {tax_results['anomaly_count']} anomalies detected")
                logger.info(f"   Mean predicted tax: ${tax_results['mean_prediction']:,.2f}")
            else:
                logger.info(f"Tax Analysis: Failed - {tax_results['error']}")

        if 'zoning_classification' in self.results:
            zoning_results = self.results['zoning_classification']
            if 'error' not in zoning_results:
                logger.info(f"Zoning Analysis: {zoning_results['low_confidence_count']} low-confidence predictions")
                logger.info(f"   Average confidence: {zoning_results['average_confidence']:.1%}")
            else:
                logger.info(f"Zoning Analysis: Failed - {zoning_results['error']}")

        if 'investment_risk' in self.results:
            risk_results = self.results['investment_risk']
            if 'error' not in risk_results:
                logger.info(f"Investment Analysis: {risk_results['high_risk_count']} high-risk properties")
                logger.info(f"   Mean risk score: {risk_results['mean_risk_score']:.3f}")
            else:
                logger.info(f"Investment Analysis: Failed - {risk_results['error']}")

if __name__ == '__main__':
    pipeline = PropertyInvestmentAnalyzerPipeline()
    if pipeline.initialize():
        if pipeline.load_property_data():
            pipeline.run_analysis()
        else:
            logger.error("Data loading failed.")
    else:
        logger.error("Model initialization failed.")
