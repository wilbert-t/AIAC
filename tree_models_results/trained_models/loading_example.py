
# Example: Loading and using saved tree-based models

import joblib
import numpy as np
import pandas as pd
import json

# Load best model and metadata
model = joblib.load('trained_models/xgboost_model.joblib')

# Load metadata
with open('trained_models/xgboost_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Loaded model: {metadata['model_name']}")
print(f"Model type: {metadata['model_type']}")
print(f"Test R²: {metadata['performance']['test_r2']:.4f}")

# Make predictions on new data
# X_new = your_new_feature_matrix  # Must have same features as training
# Feature order must match: ['mean_bond_length', 'std_bond_length', 'n_bonds', 'mean_coordination', 'std_coordination', 'max_coordination', 'radius_of_gyration', 'asphericity', 'surface_fraction', 'x_range', 'y_range', 'z_range', 'anisotropy', 'compactness', 'bond_variance', 'soap_pc_1', 'soap_pc_2', 'soap_pc_3', 'soap_pc_4', 'soap_pc_5', 'soap_pc_6', 'soap_pc_7', 'soap_pc_8', 'soap_pc_9', 'soap_pc_10', 'soap_pc_11', 'soap_pc_12', 'soap_pc_13', 'soap_pc_14', 'soap_pc_15']
# predictions = model.predict(X_new)

print("Tree-based model loaded successfully!")
