#!/usr/bin/env python3
"""
Create a mock iris dataset for testing the experiment agent
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.datasets import load_iris

def create_mock_iris_dataset():
    """Create a mock iris dataset and save it locally"""
    
    # Load the real iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(data_dir, 'iris_classification.csv')
    df.to_csv(csv_path, index=False)
    
    # Create metadata
    metadata = {
        "name": "iris_classification",
        "type": "supervised",
        "task": "classification",
        "columns": list(df.columns),
        "target": "species",
        "shape": list(df.shape),
        "classes": list(iris.target_names),
        "data_quality": "excellent",
        "missing_values": 0,
        "local_path": csv_path,
        "description": "Classic iris flower classification dataset"
    }
    
    # Save metadata
    metadata_path = os.path.join(data_dir, 'iris_classification_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created mock iris dataset:")
    print(f"  CSV: {csv_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    return csv_path, metadata_path

if __name__ == "__main__":
    create_mock_iris_dataset()