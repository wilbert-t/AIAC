#!/usr/bin/env python3
"""
Test script to demonstrate data leakage in Au cluster energy prediction
Shows the difference between models with and without leaky features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def test_data_leakage():
    """Test the effect of data leakage on model performance"""
    
    print("🔍 DATA LEAKAGE ANALYSIS FOR AU CLUSTER ENERGY PREDICTION")
    print("="*70)
    
    # Load data
    df = pd.read_csv('./au_cluster_analysis_results/descriptors.csv')
    
    print(f"📊 Dataset: {len(df)} samples")
    print(f"📊 Target range: [{df['energy'].min():.3f}, {df['energy'].max():.3f}] eV")
    print(f"📊 Target std: {df['energy'].std():.3f} eV")
    
    # Define feature sets
    basic_features = [
        'mean_bond_length', 'std_bond_length', 'n_bonds',
        'mean_coordination', 'std_coordination', 'max_coordination', 
        'surface_fraction', 'bond_variance'
    ]
    
    # LEAKY features (derived from same coordinates as energy)
    leaky_features = [
        'radius_of_gyration', 'asphericity', 'x_range', 'y_range', 'z_range',
        'soap_pc_1', 'soap_pc_2', 'soap_pc_3', 'soap_pc_4', 'soap_pc_5'
    ] + [f'rdf_bin_{i}' for i in range(1, 21)]
    
    # Get available features
    available_basic = [f for f in basic_features if f in df.columns]
    available_leaky = [f for f in leaky_features if f in df.columns]
    
    print(f"📊 Available basic features: {len(available_basic)}")
    print(f"📊 Available leaky features: {len(available_leaky)}")
    
    # Prepare data
    X_basic = df[available_basic].fillna(df[available_basic].mean())
    X_leaky = df[available_basic + available_leaky].fillna(df[available_basic + available_leaky].mean())
    y = df['energy']
    
    # Train-test split
    X_basic_train, X_basic_test, y_train, y_test = train_test_split(
        X_basic, y, test_size=0.2, random_state=42
    )
    X_leaky_train, X_leaky_test, _, _ = train_test_split(
        X_leaky, y, test_size=0.2, random_state=42
    )
    
    # Test models
    results = {}
    
    for name, X_train, X_test in [
        ("Safe Features Only", X_basic_train, X_basic_test),
        ("With Leaky Features", X_leaky_train, X_leaky_test)
    ]:
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'overfitting': train_r2 - test_r2,
            'n_features': X_train.shape[1]
        }
        
        print(f"\n📊 {name.upper()}:")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Train R²: {train_r2:.4f}")
        print(f"   Test R²:  {test_r2:.4f}")
        print(f"   Train RMSE: {train_rmse:.4f} eV")
        print(f"   Test RMSE:  {test_rmse:.4f} eV")
        print(f"   Overfitting gap: {train_r2 - test_r2:.4f}")
        
        # Assessment
        if test_r2 > 0.99:
            print(f"   ⚠️  SUSPICIOUS: R² = {test_r2:.4f} suggests data leakage!")
        elif test_r2 > 0.8:
            print(f"   ✅ GOOD: Realistic performance")
        elif test_r2 > 0.5:
            print(f"   ⚠️  MODERATE: Acceptable but could improve")
        else:
            print(f"   ❌ POOR: Low predictive power")
    
    print(f"\n" + "="*70)
    print("🎯 CONCLUSIONS:")
    
    safe_r2 = results["Safe Features Only"]["test_r2"]
    leaky_r2 = results["With Leaky Features"]["test_r2"]
    
    if leaky_r2 > 0.99 and safe_r2 < 0.9:
        print("✅ DATA LEAKAGE CONFIRMED!")
        print("   - Leaky features give unrealistic perfect scores")
        print("   - Safe features show realistic performance")
        print("   - Use safe features for honest model evaluation")
    elif safe_r2 > 0.9:
        print("✅ MODEL PERFORMANCE IS GENUINELY GOOD")
        print("   - Even safe features achieve high R²")
        print("   - Au cluster energies may be highly predictable")
    else:
        print("⚠️  CHALLENGING PREDICTION TASK")
        print("   - Low R² even with many features")
        print("   - May need more sophisticated approaches")
    
    return results

if __name__ == "__main__":
    results = test_data_leakage()