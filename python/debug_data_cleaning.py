#!/usr/bin/env python3
"""
Debug script to isolate the mixed data type comparison error in data cleaning
"""

import json
import pandas as pd
import numpy as np
from src.tools.data_tools import data_cleaning_tool

def test_data_cleaning():
    """Test data cleaning with different dataset types"""
    
    # Test case 1: Simple dataset
    print("=== Testing Simple Dataset ===")
    dataset_info = {
        "name": "test-dataset",
        "source": "test",
        "num_features": 3
    }
    
    try:
        result = data_cleaning_tool(json.dumps(dataset_info), sample_size=100)
        result_data = json.loads(result)
        if "error" in result_data:
            print(f"ERROR: {result_data['error']}")
        else:
            print("SUCCESS: Simple dataset cleaned successfully")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
    
    # Test case 2: Sentiment dataset
    print("\n=== Testing Sentiment Dataset ===")
    dataset_info = {
        "name": "sentiment-analysis-dataset",
        "source": "test",
        "num_features": 4
    }
    
    try:
        result = data_cleaning_tool(json.dumps(dataset_info), sample_size=100)
        result_data = json.loads(result)
        if "error" in result_data:
            print(f"ERROR: {result_data['error']}")
        else:
            print("SUCCESS: Sentiment dataset cleaned successfully")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
    
    # Test case 3: Customer support dataset (the problematic one)
    print("\n=== Testing Customer Support Dataset ===")
    dataset_info = {
        "name": "suraj520/customer-support-ticket-dataset",
        "source": "Kaggle",
        "num_features": 5
    }
    
    try:
        result = data_cleaning_tool(json.dumps(dataset_info), sample_size=100)
        result_data = json.loads(result)
        if "error" in result_data:
            print(f"ERROR: {result_data['error']}")
        else:
            print("SUCCESS: Customer support dataset cleaned successfully")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")

if __name__ == "__main__":
    test_data_cleaning()