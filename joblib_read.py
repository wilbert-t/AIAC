import joblib
import numpy as np

def inspect_joblib(path):
    print(f"🔍 Inspecting: {path}\n")
    obj = joblib.load(path)

    # Print main type
    print("Object type:", type(obj))

    # If it's a dictionary
    if isinstance(obj, dict):
        print("\n📦 Dictionary with keys:")
        for key, value in obj.items():
            print(f"  - {key}: {type(value)}")
    
    # If it's a list
    elif isinstance(obj, list):
        print(f"\n📦 List with {len(obj)} items")
        if len(obj) > 0:
            print("  First item type:", type(obj[0]))
    
    # If it's a numpy array
    elif isinstance(obj, np.ndarray):
        print(f"\n📊 Numpy array with shape {obj.shape}, dtype={obj.dtype}")
        print("First 5 entries:", obj[:5])
    
    # If it's a scikit-learn model
    elif hasattr(obj, "get_params"):
        print("\n🤖 Looks like a scikit-learn model!")
        print("Model parameters:")
        print(obj.get_params())
    
    else:
        print("\nℹ️ Object details:")
        print(obj)

    return obj

# 👉 Replace with your actual filename:
my_obj = inspect_joblib("/Users/wilbert/Documents/GitHub/AIAC/linear_models_results/elastic_net_model.joblib")