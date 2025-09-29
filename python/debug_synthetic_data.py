#!/usr/bin/env python3
"""
Debug script to isolate the synthetic data generation issue
"""

import json
import pandas as pd
import numpy as np
from src.tools.data_tools import _generate_synthetic_data, _assess_data_quality, _clean_data

def test_synthetic_data_generation():
    """Test synthetic data generation with different dataset patterns"""
    
    # Test datasets that are failing
    failing_datasets = [
        {
            "name": "suraj520/customer-support-ticket-dataset",
            "source": "Kaggle",
            "num_features": 5
        },
        {
            "name": "customer-support-ticket-dataset",
            "source": "Kaggle", 
            "num_features": 5
        },
        {
            "name": "support-ticket-data",
            "source": "Kaggle",
            "num_features": 5
        }
    ]
    
    # Test datasets that are working
    working_datasets = [
        {
            "name": "kanchana1990/uber-customer-reviews-dataset-2024",
            "source": "Kaggle",
            "num_features": 6
        },
        {
            "name": "sentiment-analysis-data",
            "source": "Kaggle",
            "num_features": 4
        }
    ]
    
    all_datasets = failing_datasets + working_datasets
    
    for dataset in all_datasets:
        print(f"\n=== Testing Dataset: {dataset['name']} ===")
        
        try:
            # Step 1: Generate synthetic data
            print("Step 1: Generating synthetic data...")
            synthetic_data = _generate_synthetic_data(dataset, 1000)
            print(f"SUCCESS: Generated data shape: {synthetic_data.shape}")
            print(f"Data types: {synthetic_data.dtypes.to_dict()}")
            print(f"Sample data:\n{synthetic_data.head()}")
            
            # Step 2: Assess data quality
            print("\nStep 2: Assessing data quality...")
            quality_metrics = _assess_data_quality(synthetic_data)
            print(f"SUCCESS: Quality metrics: {quality_metrics}")
            
            # Step 3: Clean data
            print("\nStep 3: Cleaning data...")
            cleaned_data, preprocessing_steps = _clean_data(synthetic_data)
            print(f"SUCCESS: Cleaned data shape: {cleaned_data.shape}")
            print(f"Preprocessing steps: {len(preprocessing_steps)}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    test_synthetic_data_generation()