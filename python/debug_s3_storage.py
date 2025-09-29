#!/usr/bin/env python3
"""
Debug script to isolate the mixed data type comparison error in S3 storage
"""

import json
from src.tools.data_tools import data_cleaning_tool, s3_storage_tool

def test_s3_storage():
    """Test S3 storage with cleaned data"""
    
    print("=== Testing S3 Storage Pipeline ===")
    
    # Step 1: Clean data
    dataset_info = {
        "name": "suraj520/customer-support-ticket-dataset",
        "source": "Kaggle",
        "num_features": 5
    }
    
    print("Step 1: Cleaning data...")
    try:
        cleaning_result = data_cleaning_tool(json.dumps(dataset_info), sample_size=100)
        cleaning_data = json.loads(cleaning_result)
        if "error" in cleaning_data:
            print(f"CLEANING ERROR: {cleaning_data['error']}")
            return
        else:
            print("SUCCESS: Data cleaned successfully")
    except Exception as e:
        print(f"CLEANING EXCEPTION: {str(e)}")
        return
    
    # Step 2: Store in S3
    print("\nStep 2: Storing in S3...")
    try:
        storage_result = s3_storage_tool(
            json.dumps(dataset_info), 
            cleaning_result, 
            query="analyze customer sentiment data"
        )
        storage_data = json.loads(storage_result)
        if "error" in storage_data:
            print(f"STORAGE ERROR: {storage_data['error']}")
        else:
            print("SUCCESS: Data stored successfully")
    except Exception as e:
        print(f"STORAGE EXCEPTION: {str(e)}")

if __name__ == "__main__":
    test_s3_storage()