#!/usr/bin/env python3
"""
Debug script to test specific datasets that are failing in the integration test
"""

import json
from src.tools.data_tools import data_cleaning_tool

def test_specific_datasets():
    """Test the specific datasets that are failing"""
    
    # These are the datasets from the integration test
    failing_datasets = [
        {
            "name": "suraj520/customer-support-ticket-dataset",
            "source": "Kaggle",
            "title": "Customer Support Ticket Dataset",
            "description": "Customer support tickets with categories",
            "num_features": 5
        },
        {
            "name": "kanchana1990/uber-customer-reviews-dataset-2024", 
            "source": "Kaggle",
            "title": "Uber Customer Reviews Dataset 2024",
            "description": "Customer reviews for Uber services",
            "num_features": 6
        }
    ]
    
    for i, dataset in enumerate(failing_datasets, 1):
        print(f"\n=== Testing Dataset {i}: {dataset['name']} ===")
        
        # Test with different sample sizes
        for sample_size in [100, 500, 1000]:
            print(f"\nTesting with sample_size={sample_size}")
            try:
                result = data_cleaning_tool(json.dumps(dataset), sample_size=sample_size)
                result_data = json.loads(result)
                if "error" in result_data:
                    print(f"ERROR (size {sample_size}): {result_data['error']}")
                    # Try to get more details about the error
                    import traceback
                    try:
                        # Re-run to get full traceback
                        data_cleaning_tool(json.dumps(dataset), sample_size=sample_size)
                    except Exception as e:
                        print(f"Full traceback: {traceback.format_exc()}")
                    break
                else:
                    print(f"SUCCESS (size {sample_size}): Cleaned successfully")
            except Exception as e:
                print(f"EXCEPTION (size {sample_size}): {str(e)}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                break

if __name__ == "__main__":
    test_specific_datasets()