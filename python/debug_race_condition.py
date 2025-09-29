#!/usr/bin/env python3
"""
Debug script to test for race conditions or state issues in data cleaning
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from src.tools.data_tools import data_cleaning_tool

def test_concurrent_cleaning():
    """Test data cleaning with concurrent execution to check for race conditions"""
    
    # Test the problematic dataset multiple times
    dataset_info = {
        "name": "suraj520/customer-support-ticket-dataset",
        "source": "Kaggle",
        "num_features": 5
    }
    
    def clean_dataset(iteration):
        """Clean dataset in a separate thread"""
        try:
            result = data_cleaning_tool(json.dumps(dataset_info), sample_size=1000)
            result_data = json.loads(result)
            if "error" in result_data:
                return f"Iteration {iteration}: ERROR - {result_data['error']}"
            else:
                return f"Iteration {iteration}: SUCCESS"
        except Exception as e:
            return f"Iteration {iteration}: EXCEPTION - {str(e)}"
    
    print("=== Testing Sequential Execution ===")
    for i in range(10):
        result = clean_dataset(i)
        print(result)
        time.sleep(0.1)  # Small delay
    
    print("\n=== Testing Concurrent Execution ===")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(clean_dataset, i) for i in range(10)]
        for future in futures:
            print(future.result())
    
    print("\n=== Testing Different Dataset Names ===")
    test_datasets = [
        {"name": "suraj520/customer-support-ticket-dataset", "source": "Kaggle", "num_features": 5},
        {"name": "customer-support-dataset", "source": "Kaggle", "num_features": 5},
        {"name": "support-ticket-data", "source": "Kaggle", "num_features": 5},
        {"name": "sentiment-analysis-data", "source": "Kaggle", "num_features": 5},
        {"name": "text-classification-data", "source": "Kaggle", "num_features": 5}
    ]
    
    for dataset in test_datasets:
        try:
            result = data_cleaning_tool(json.dumps(dataset), sample_size=1000)
            result_data = json.loads(result)
            if "error" in result_data:
                print(f"Dataset '{dataset['name']}': ERROR - {result_data['error']}")
            else:
                print(f"Dataset '{dataset['name']}': SUCCESS")
        except Exception as e:
            print(f"Dataset '{dataset['name']}': EXCEPTION - {str(e)}")

if __name__ == "__main__":
    test_concurrent_cleaning()