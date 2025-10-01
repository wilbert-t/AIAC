
# Example: Loading and using saved models

import joblib
import numpy as np
import pandas as pd

# Load best model and scaler
model = joblib.load('trained_models/svr_linear_model.joblib')
scaler = joblib.load('trained_models/svr_linear_scaler.joblib')

# Make predictions on new data
# X_new = your_new_feature_matrix  # Must have same features as training
# X_new_scaled = scaler.transform(X_new)
# predictions = model.predict(X_new_scaled)

print("Models loaded successfully!")
