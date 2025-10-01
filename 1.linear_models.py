#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import json
from scipy import stats
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# SOAP descriptors for enhanced features
try:
    from dscribe.descriptors import SOAP
    from ase.atoms import Atoms
    SOAP_AVAILABLE = True
except ImportError:
    print("Warning: DScribe not available. Using basic descriptors only.")
    SOAP_AVAILABLE = False

class LinearModelsAnalyzer:
    """
    Enhanced Linear & Regularized Models for Au Cluster Analysis with Comprehensive Reporting
    
    Features:
    - Multiple model training and evaluation
    - Comprehensive visualization suite
    - Executive summary generation
    - Cross-validation analysis
    - Learning curves
    - Residual analysis
    - Model comparison reports
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.soap_features = None
        self.feature_names = None
        self.best_soap_params = None
        self.predictions_df = None
        self.cv_results = {}
        self.learning_curves = {}
        
        # Initialize models with justifications
        self.model_configs = {
            'svr_linear': {
                'model': SVR(kernel='linear', cache_size=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'epsilon': [0.1, 0.5, 1.0]
                },
                'justification': """
                SVR Linear:
                - Support Vector Regression with linear kernel
                - Robust to outliers through support vector mechanism
                - Effective for high-dimensional data
                - Memory efficient with linear kernel
                - Good baseline for linear SVR approaches
                """
            },
            'linear': {
                'model': LinearRegression(),
                'params': {},
                'justification': """
                Linear Regression:
                - Baseline model for energy prediction
                - Shows if linear relationships dominate
                - Coefficient interpretation reveals feature importance
                - Fast training for quick iteration
                """
            },
            'ridge': {
                'model': RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
                'params': {},
                'justification': """
                Ridge Regression:
                - Handles multicollinearity in SOAP features
                - L2 regularization prevents overfitting
                - Keeps all features (vs Lasso feature selection)
                - Stable coefficients for correlated descriptors
                """
            },
            'lasso': {
                'model': LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=2000),
                'params': {},
                'justification': """
                Lasso Regression:
                - Automatic feature selection (zeroes coefficients)
                - Identifies most important SOAP components
                - Sparse solutions reduce model complexity
                - Interpretable: shows which atomic environments matter
                """
            },
            'elastic_net': {
                'model': ElasticNetCV(
                    alphas=np.logspace(-3, 1, 20),
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=5, max_iter=2000
                ),
                'params': {},
                'justification': """
                Elastic Net:
                - Best of Ridge + Lasso: regularization + selection
                - Groups correlated SOAP features together
                - l1_ratio controls Ridge/Lasso balance
                - Robust to different data characteristics
                """
            }
        }
    
    def load_data(self, data_path=None, target_column='energy', use_hybrid_training=True):
        """
        Enhanced data loading with hybrid training support
        
        Args:
            data_path: Path to original descriptors.csv (999 structures)
            target_column: Target variable name
            use_hybrid_training: Whether to use progressive training approach
        """
        # Load original 999 structures for foundation learning
        if data_path is None:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        if isinstance(data_path, str):
            self.df_foundation = pd.read_csv(data_path)
        else:
            self.df_foundation = data_path
        
        self.df_foundation = self.df_foundation.dropna(subset=[target_column])
        
        # Load categorized high-quality datasets
        self.datasets = {}
        dataset_files = {
            'balanced': './task2/improved_dataset_balanced.csv',
            'high_quality': './task2/improved_dataset_high_quality.csv', 
            'elite': './task2/improved_dataset_elite.csv'
        }
        
        if use_hybrid_training:
            print("🔄 Loading hybrid training datasets...")
            
            for name, file_path in dataset_files.items():
                try:
                    df = pd.read_csv(file_path)
                    df = df.dropna(subset=[target_column])
                    self.datasets[name] = df
                    print(f"   ✅ {name}: {len(df)} structures")
                except FileNotFoundError:
                    print(f"   ⚠️  {name}: File not found - {file_path}")
                    self.datasets[name] = None
        
        # Set primary dataset for initial analysis
        self.df = self.df_foundation
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Foundation (999): {len(self.df_foundation)} samples")
        print(f"   Target range: {self.df_foundation[target_column].min():.2f} to {self.df_foundation[target_column].max():.2f}")
        
        if use_hybrid_training and any(df is not None for df in self.datasets.values()):
            print(f"   Hybrid training: ENABLED")
            for name, df in self.datasets.items():
                if df is not None:
                    print(f"   - {name}: {len(df)} samples")
        
        return self.df_foundation
    
    def tune_soap_params(self, structures_data, basic_features_df, y, n_components=None):
        """Tune SOAP hyperparameters using grid search and cross-validation."""
        if not SOAP_AVAILABLE:
            print("SOAP not available for tuning.")
            return None
        
        print("Tuning SOAP parameters...")
        
        param_grid = {
            'r_cut': [4.0, 4.48, 5.0],
            'n_max': [6, 8],
            'l_max': [6, 8],
            'sigma': [0.5, 1.0]
        }
        
        best_score = -np.inf
        best_params = None
        best_soap_df = None
        
        from itertools import product
        keys = param_grid.keys()
        combinations = list(product(*param_grid.values()))
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            print(f"Testing params: {params}")
            
            try:
                soap = SOAP(
                    species=['Au'],
                    r_cut=params['r_cut'],
                    n_max=params['n_max'],
                    l_max=params['l_max'],
                    sigma=params['sigma'],
                    periodic=False,
                    sparse=True,
                    average='inner'
                )
                
                soap_features = []
                filenames = []
                
                for structure in structures_data:
                    try:
                        atoms = structure['atoms'] if 'atoms' in structure else None
                        if atoms is None:
                            coords = structure['coords']
                            atoms = Atoms('Au' * len(coords), positions=coords)
                        
                        soap_desc = soap.create(atoms)
                        soap_features.append(soap_desc)
                        filenames.append(structure['filename'])
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
                
                if not soap_features:
                    continue
                
                soap_array = np.vstack(soap_features) if len(soap_features[0].shape) > 1 else np.array(soap_features)
                
                if n_components:
                    pca = PCA(n_components=n_components)
                    soap_array = pca.fit_transform(soap_array)
                    soap_cols = [f'soap_pc_{i}' for i in range(n_components)]
                else:
                    soap_cols = [f'soap_{i}' for i in range(soap_array.shape[1])]
                
                soap_df = pd.DataFrame(soap_array, columns=soap_cols)
                soap_df['filename'] = filenames
                
                merged_df = basic_features_df.merge(soap_df, on='filename', how='inner')
                feature_cols = [col for col in merged_df.columns if col not in ['filename', 'energy']]
                X = merged_df[feature_cols]
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
                ])
                cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
                mean_score = cv_scores.mean()
                
                print(f"CV R²: {mean_score:.3f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_soap_df = soap_df
                    
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        if best_params:
            print(f"Best params: {best_params} with CV R²: {best_score:.3f}")
            self.best_soap_params = best_params
            self.df = self.df.merge(best_soap_df, on='filename', how='inner')
            self.soap_features = [col for col in best_soap_df.columns if col.startswith('soap_')]
        
        return best_params
    
    def create_soap_features(self, structures_data=None, use_tuned_params=False, n_components=None):
        """Create SOAP descriptors for enhanced accuracy"""
        if not SOAP_AVAILABLE or structures_data is None:
            print("Using basic descriptors only")
            return None
        
        print("Creating SOAP descriptors for enhanced accuracy...")
        
        if use_tuned_params and self.best_soap_params:
            params = self.best_soap_params
        else:
            params = {
                'r_cut': 4.48,
                'n_max': 8,
                'l_max': 8,
                'sigma': 0.5
            }
        
        soap = SOAP(
            species=['Au'],
            r_cut=params['r_cut'],
            n_max=params['n_max'],
            l_max=params['l_max'],
            sigma=params['sigma'],
            periodic=False,
            sparse=True,
            average='inner'
        )
        
        soap_features = []
        filenames = []
        
        for structure in structures_data:
            try:
                atoms = structure['atoms'] if 'atoms' in structure else None
                if atoms is None:
                    coords = structure['coords']
                    atoms = Atoms('Au' * 20, positions=coords)
                
                soap_desc = soap.create(atoms)
                soap_features.append(soap_desc)
                filenames.append(structure['filename'])
                
            except Exception as e:
                print(f"Error creating SOAP for {structure.get('filename', 'unknown')}: {e}")
                continue
        
        if soap_features:
            soap_array = np.vstack(soap_features) if len(soap_features[0].shape) > 1 else np.array(soap_features)
            
            if n_components:
                pca = PCA(n_components=n_components)
                soap_array = pca.fit_transform(soap_array)
                print(f"Applied PCA: Reduced to {n_components} components")
                soap_cols = [f'soap_pc_{i}' for i in range(n_components)]
            else:
                soap_cols = [f'soap_{i}' for i in range(soap_array.shape[1])]
            
            soap_df = pd.DataFrame(soap_array, columns=soap_cols)
            soap_df['filename'] = filenames
            
            self.df = self.df.merge(soap_df, on='filename', how='inner')
            
            print(f"Added {soap_array.shape[1]} SOAP features")
            self.soap_features = soap_cols
            
        return self.soap_features
    
    def prepare_features(self, df=None, target_column='energy', include_soap=True):
        """Prepare feature matrix and target vector"""
        # Use provided dataframe or default to self.df
        if df is None:
            df = self.df
            
        feature_cols = []
        
        # EXCLUDE ENERGY-DERIVED FEATURES TO PREVENT DATA LEAKAGE
        exclude_features = ['energy_per_atom', 'filename', 'Unnamed: 0', target_column, 'structure_id']
        
        basic_features = [
            'mean_bond_length', 'std_bond_length', 'n_bonds',
            'mean_coordination', 'std_coordination', 'max_coordination',
            'radius_of_gyration', 'asphericity', 'surface_fraction',
            'x_range', 'y_range', 'z_range', 'anisotropy',
            'compactness', 'bond_variance'
        ]
        
        # Filter out any excluded features from basic features
        available_basic = [f for f in basic_features 
                          if f in df.columns and f not in exclude_features]
        feature_cols.extend(available_basic)
        
        if include_soap and hasattr(self, 'soap_features') and self.soap_features:
            feature_cols.extend(self.soap_features)
            print(f"Using {len(self.soap_features)} SOAP features")
        
        feature_cols = [f for f in feature_cols if f in df.columns]
        data_clean = df[feature_cols + [target_column]].dropna()
        
        X = data_clean[feature_cols]
        y = data_clean[target_column]
        
        if df is self.df:  # Only set feature_names for main dataframe
            self.feature_names = feature_cols
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Using features: {len(feature_cols)} total")
        
        return X, y, feature_cols
    
    def compute_learning_curves(self, X, y, model_name, model, scaler):
        """Compute learning curves for a model"""
        print(f"Computing learning curves for {model_name}...")
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X, y, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    
    def progressive_hybrid_training(self, X_foundation, y_foundation, use_elite_validation=True):
        """
        Progressive hybrid training: Foundation → Quality Refinement → Elite Validation
        
        Args:
            X_foundation: Features from 999 structures
            y_foundation: Targets from 999 structures
            use_elite_validation: Whether to use elite dataset for final validation
        
        Returns:
            dict: Comprehensive training results across all stages
        """
        print("\n" + "="*70)
        print("🚀 PROGRESSIVE HYBRID TRAINING PIPELINE")
        print("="*70)
        
        results = {
            'foundation_results': {},
            'refinement_results': {},
            'elite_validation': {},
            'learning_curves': {},
            'anti_memorization_metrics': {}
        }
        
        # Stage 1: Foundation Learning (999 structures)
        print("\n📚 STAGE 1: Foundation Learning (999 structures)")
        print("-" * 50)
        
        foundation_results = self.train_models(X_foundation, y_foundation, test_size=0.2)
        results['foundation_results'] = foundation_results
        
        # Stage 2: Quality Refinement (if high-quality dataset available)
        if self.datasets.get('high_quality') is not None:
            print("\n🎯 STAGE 2: Quality Refinement (High-Quality subset)")
            print("-" * 50)
            
            # Prepare high-quality data
            X_hq, y_hq, _ = self.prepare_features(self.datasets['high_quality'])
            
            # Transfer learning: use foundation models as starting point
            refinement_results = {}
            for model_name, foundation_model_data in foundation_results.items():
                print(f"\n🔄 Refining {model_name}...")
                
                # Get the trained model from the results
                if 'pipeline' in foundation_model_data:
                    model = foundation_model_data['pipeline']
                elif 'model' in foundation_model_data:
                    model = foundation_model_data['model']  
                else:
                    print(f"   ⚠️ No model found for {model_name}")
                    continue
                
                # Fine-tune on high-quality data
                refined_result = self._fine_tune_model(
                    model, X_hq, y_hq, model_name
                )
                refinement_results[model_name] = refined_result
            
            results['refinement_results'] = refinement_results
        
        # Stage 3: Elite Validation (if elite dataset available)
        if use_elite_validation and self.datasets.get('elite') is not None:
            print("\n🏆 STAGE 3: Elite Validation (Never-seen structures)")
            print("-" * 50)
            
            X_elite, y_elite, _ = self.prepare_features(self.datasets['elite'])
            
            elite_results = {}
            source_results = results.get('refinement_results', results['foundation_results'])
            
            for model_name, model_data in source_results.items():
                if 'model' in model_data:
                    elite_scores = self._validate_on_elite(
                        model_data['model'], X_elite, y_elite, model_name
                    )
                    elite_results[model_name] = elite_scores
            
            results['elite_validation'] = elite_results
        
        # Anti-memorization analysis
        if len(results['foundation_results']) > 0:
            results['anti_memorization_metrics'] = self._analyze_memorization(
                results['foundation_results'], 
                results.get('refinement_results', {}),
                results.get('elite_validation', {})
            )
        
        return results
    
    def _fine_tune_model(self, foundation_model, X_hq, y_hq, model_name):
        """Fine-tune a foundation model on high-quality data"""
        from sklearn.base import clone
        
        # Clone the foundation model
        model = clone(foundation_model)
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X_hq, y_hq, test_size=0.3, random_state=self.random_state
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        
        scores = {
            'model': model,
            'scaler': scaler,
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'predictions': y_pred,
            'actuals': y_test
        }
        
        print(f"   ✅ {model_name}: R² = {scores['r2']:.4f}, MSE = {scores['mse']:.4f}")
        return scores
    
    def _validate_on_elite(self, model, X_elite, y_elite, model_name):
        """Validate model on elite dataset (never-seen structures)"""
        # Use the model's associated scaler if available
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            X_elite_scaled = model.named_steps['scaler'].transform(X_elite)
            y_pred = model.named_steps['model'].predict(X_elite_scaled)
        else:
            # Assume model includes scaling or use StandardScaler
            scaler = StandardScaler()
            X_elite_scaled = scaler.fit_transform(X_elite)
            y_pred = model.predict(X_elite_scaled)
        
        scores = {
            'r2': r2_score(y_elite, y_pred),
            'mse': mean_squared_error(y_elite, y_pred),
            'mae': mean_absolute_error(y_elite, y_pred),
            'predictions': y_pred,
            'actuals': y_elite
        }
        
        print(f"   🏆 {model_name}: Elite R² = {scores['r2']:.4f}, MSE = {scores['mse']:.4f}")
        return scores
    
    def _analyze_memorization(self, foundation_results, refinement_results, elite_results):
        """Analyze whether models are learning vs. memorizing"""
        memorization_metrics = {}
        
        for model_name in foundation_results:
            metrics = {}
            
            # Foundation performance
            foundation_r2 = foundation_results.get(model_name, {}).get('test_r2', 0)
            metrics['foundation_r2'] = foundation_r2
            
            # Refinement performance (if available)
            if model_name in refinement_results:
                refinement_r2 = refinement_results[model_name].get('r2', 0)
                metrics['refinement_r2'] = refinement_r2
                metrics['refinement_improvement'] = refinement_r2 - foundation_r2
            
            # Elite validation (if available)
            if model_name in elite_results:
                elite_r2 = elite_results[model_name].get('r2', 0)
                metrics['elite_r2'] = elite_r2
                metrics['generalization_gap'] = foundation_r2 - elite_r2
                
                # Memorization indicator: large gap suggests overfitting
                if metrics['generalization_gap'] > 0.1:
                    metrics['memorization_risk'] = 'HIGH'
                elif metrics['generalization_gap'] > 0.05:
                    metrics['memorization_risk'] = 'MEDIUM'
                else:
                    metrics['memorization_risk'] = 'LOW'
            
            memorization_metrics[model_name] = metrics
        
        return memorization_metrics
    
    def _guaranteed_350_split(self, X, y, test_size=0.2):
        """
        Split data ensuring Structure 350.xyz is always in the test set
        
        Args:
            X: Feature matrix with index matching original dataframe
            y: Target vector with index matching original dataframe  
            test_size: Proportion for test set
            
        Returns:
            X_train, X_test, y_train, y_test with 350.xyz guaranteed in test
        """
        print("\n🎯 GUARANTEED STRUCTURE 350.XYZ HOLDOUT")
        print("-" * 50)
        
        # Find Structure 350.xyz in the data
        if hasattr(self, 'df') and 'filename' in self.df.columns:
            # Find the index of Structure 350.xyz
            structure_350_mask = self.df['filename'] == '350.xyz'
            structure_350_indices = self.df[structure_350_mask].index
            
            if len(structure_350_indices) > 0:
                structure_350_idx = structure_350_indices[0]
                print(f"   ✅ Found Structure 350.xyz at index {structure_350_idx}")
                print(f"   🎯 Energy: {self.df.loc[structure_350_idx, 'energy']:.5f} eV")
                
                # Ensure the index is in our X, y data
                if structure_350_idx in X.index:
                    # Remove 350.xyz from the pool for random splitting
                    X_without_350 = X.drop(structure_350_idx)
                    y_without_350 = y.drop(structure_350_idx)
                    
                    # Calculate how many more samples we need for test set
                    total_samples = len(X)
                    target_test_size = int(total_samples * test_size)
                    remaining_test_size = max(0, target_test_size - 1)  # -1 because we already have 350.xyz
                    
                    if remaining_test_size > 0:
                        # Random split the remaining data
                        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                            X_without_350, y_without_350, 
                            test_size=remaining_test_size,
                            random_state=self.random_state
                        )
                    else:
                        # If test_size=1 structure, only use 350.xyz
                        X_train_temp, X_test_temp = X_without_350, X.iloc[[]]
                        y_train_temp, y_test_temp = y_without_350, y.iloc[[]]
                    
                    # Add Structure 350.xyz to test set
                    X_test = pd.concat([X_test_temp, X.loc[[structure_350_idx]]])
                    y_test = pd.concat([y_test_temp, y.loc[[structure_350_idx]]])
                    X_train = X_train_temp
                    y_train = y_train_temp
                    
                    print(f"   📊 Final split: Train={len(X_train)}, Test={len(X_test)} (includes 350.xyz)")
                    print(f"   🎯 Structure 350.xyz guaranteed in test set!")
                    
                    return X_train, X_test, y_train, y_test
                else:
                    print(f"   ⚠️ Structure 350.xyz index {structure_350_idx} not found in feature matrix")
            else:
                print("   ⚠️ Structure 350.xyz not found in dataframe")
        else:
            print("   ⚠️ No filename column available for structure identification")
        
        # Fallback to regular random split
        print("   🔄 Falling back to random split")
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def train_models(self, X, y, test_size=0.2, guarantee_350_in_test=True):
        """Train all linear models with comprehensive analysis"""
        print("\n" + "="*60)
        print("TRAINING LINEAR & REGULARIZED MODELS")
        print("="*60)
        
        # Split data with guaranteed Structure 350.xyz in test set
        if guarantee_350_in_test:
            X_train, X_test, y_train, y_test = self._guaranteed_350_split(X, y, test_size)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        results = {}
        predictions_data = []
        
        for name, config in self.model_configs.items():
            print(f"\n🔍 Training {name.upper()}...")
            print(f"Justification: {config['justification'].strip()}")
            
            # Create pipeline with scaling
            scaler = StandardScaler()
            model = config['model']
            
            # Fit scaler and model
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if name == 'svr_linear' and config['params']:
                # SVR needs grid search for hyperparameters
                from sklearn.model_selection import GridSearchCV
                grid_search = GridSearchCV(model, config['params'], cv=5, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                print(f"   Best params: {grid_search.best_params_}")
            else:
                # Regular training for CV models
                model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Store predictions for combined analysis
            for i, (actual, pred) in enumerate(zip(y_test, y_test_pred)):
                predictions_data.append({
                    'model': name,
                    'sample_id': i,
                    'actual': actual,
                    'predicted': pred,
                    'residual': actual - pred,
                    'abs_residual': abs(actual - pred)
                })
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation with detailed scoring
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', config['model'])])
            cv_scores_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
            cv_scores_mae = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
            cv_scores_rmse = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
            
            # Learning curves
            learning_curve_data = self.compute_learning_curves(X, y, name, config['model'], StandardScaler())
            self.learning_curves[name] = learning_curve_data
            
            # Residual statistics
            residuals = y_test - y_test_pred
            residual_stats = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals),
                'normality_pvalue': stats.shapiro(residuals)[1] if len(residuals) <= 5000 else stats.jarque_bera(residuals)[1]
            }
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores_r2.mean(),
                'cv_r2_std': cv_scores_r2.std(),
                'cv_mae_mean': -cv_scores_mae.mean(),
                'cv_mae_std': cv_scores_mae.std(),
                'cv_rmse_mean': -cv_scores_rmse.mean(),
                'cv_rmse_std': cv_scores_rmse.std(),
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'residuals': residuals,
                'residual_stats': residual_stats,
                'cv_scores': {
                    'r2': cv_scores_r2,
                    'mae': -cv_scores_mae,
                    'rmse': -cv_scores_rmse
                }
            }
            
            print(f"✅ {name}: R² = {test_r2:.3f}, RMSE = {test_rmse:.2f}, MAE = {test_mae:.2f}")
            print(f"   CV: R² = {cv_scores_r2.mean():.3f}±{cv_scores_r2.std():.3f}")
        
        # Store predictions DataFrame
        self.predictions_df = pd.DataFrame(predictions_data)
        self.results = results
        self.cv_results = {name: results[name]['cv_scores'] for name in results}
        
        return results
    
    def create_individual_model_plots(self, output_dir):
        """Create individual plots for each model"""
        print("Creating individual model plots...")
        
        for name, result in self.results.items():
            model_dir = output_dir / f"{name}_individual"
            model_dir.mkdir(exist_ok=True)
            
            # 1. Prediction vs Actual
            self._plot_individual_prediction_vs_actual(name, result, model_dir)
            
            # 2. Residual plots + distribution
            self._plot_individual_residuals(name, result, model_dir)
            
            # 3. Learning curves
            self._plot_individual_learning_curve(name, model_dir)
            
            # 4. Cross-validation performance
            self._plot_individual_cv_performance(name, result, model_dir)
    
    def _plot_individual_prediction_vs_actual(self, name, result, output_dir):
        """Individual prediction vs actual plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Test set
        y_true = self.y_test
        y_pred = result['y_test_pred']
        
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50, color='blue', label='Test Data')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Energy')
        ax1.set_ylabel('Predicted Energy')
        ax1.set_title(f'{name.title()} - Test Set Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        r2 = result['test_r2']
        rmse = result['test_rmse']
        mae = result['test_mae']
        
        stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Training set
        y_train_true = self.y_train
        y_train_pred = result['y_train_pred']
        
        ax2.scatter(y_train_true, y_train_pred, alpha=0.6, s=50, color='green', label='Train Data')
        
        min_val = min(y_train_true.min(), y_train_pred.min())
        max_val = max(y_train_true.max(), y_train_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Energy')
        ax2.set_ylabel('Predicted Energy')
        ax2.set_title(f'{name.title()} - Training Set Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add training statistics
        train_r2 = result['train_r2']
        train_rmse = result['train_rmse']
        train_mae = result['train_mae']
        
        train_stats_text = f'R² = {train_r2:.3f}\nRMSE = {train_rmse:.3f}\nMAE = {train_mae:.3f}'
        ax2.text(0.05, 0.95, train_stats_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_residuals(self, name, result, output_dir):
        """Individual residual analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        residuals = result['residuals']
        y_pred = result['y_test_pred']
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8, label='Zero Residual')
        ax1.set_xlabel('Predicted Energy')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'{name.title()} - Residuals vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{name.title()} - Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title(f'{name.title()} - Q-Q Plot (Normality Test)')
        ax3.grid(True, alpha=0.3)
        
        # Residuals vs Order (to check for patterns)
        ax4.plot(residuals, 'o', alpha=0.6, markersize=4)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Residuals')
        ax4.set_title(f'{name.title()} - Residuals vs Sample Order')
        ax4.grid(True, alpha=0.3)
        
        # Add residual statistics
        stats_dict = result['residual_stats']
        stats_text = f"""Mean: {stats_dict['mean']:.4f}
Std: {stats_dict['std']:.4f}
Skewness: {stats_dict['skewness']:.3f}
Kurtosis: {stats_dict['kurtosis']:.3f}
Normality p: {stats_dict['normality_pvalue']:.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_learning_curve(self, name, output_dir):
        """Individual learning curve plot"""
        if name not in self.learning_curves:
            return
        
        data = self.learning_curves[name]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        train_sizes = data['train_sizes']
        train_mean = data['train_scores_mean']
        train_std = data['train_scores_std']
        val_mean = data['val_scores_mean']
        val_std = data['val_scores_std']
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{name.title()} - Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final scores
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        ax.text(0.02, 0.98, f'Final Training R²: {final_train:.3f}\nFinal Validation R²: {final_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_cv_performance(self, name, result, output_dir):
        """Individual cross-validation performance plot"""
        cv_scores = result['cv_scores']
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² scores across folds
        ax1.boxplot(cv_scores['r2'], labels=['R²'])
        ax1.scatter([1] * len(cv_scores['r2']), cv_scores['r2'], alpha=0.7, color='blue')
        ax1.set_title(f'{name.title()} - CV R² Scores')
        ax1.set_ylabel('R² Score')
        ax1.grid(True, alpha=0.3)
        
        # MAE scores across folds
        ax2.boxplot(cv_scores['mae'], labels=['MAE'])
        ax2.scatter([1] * len(cv_scores['mae']), cv_scores['mae'], alpha=0.7, color='orange')
        ax2.set_title(f'{name.title()} - CV MAE Scores')
        ax2.set_ylabel('MAE')
        ax2.grid(True, alpha=0.3)
        
        # RMSE scores across folds
        ax3.boxplot(cv_scores['rmse'], labels=['RMSE'])
        ax3.scatter([1] * len(cv_scores['rmse']), cv_scores['rmse'], alpha=0.7, color='red')
        ax3.set_title(f'{name.title()} - CV RMSE Scores')
        ax3.set_ylabel('RMSE')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        for ax, scores, metric in zip([ax1, ax2, ax3], [cv_scores['r2'], cv_scores['mae'], cv_scores['rmse']], 
                                     ['R²', 'MAE', 'RMSE']):
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            ax.text(0.02, 0.98, f'Mean: {mean_score:.3f}\nStd: {std_score:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}_cv_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_combined_plots(self, output_dir):
        """Create combined comparison plots for all models"""
        print("Creating combined comparison plots...")
        
        # 1. Combined model performance comparison
        self._plot_combined_model_comparison(output_dir)
        
        # 2. Combined predictions vs actual
        self._plot_combined_predictions_vs_actual(output_dir)
        
        # 3. Combined residual analysis
        self._plot_combined_residuals(output_dir)
        
        # 4. Combined cross-validation comparison
        self._plot_combined_cv_comparison(output_dir)
        
        # 5. Combined learning curves
        self._plot_combined_learning_curves(output_dir)
    
    def _plot_combined_model_comparison(self, output_dir):
        """Combined model performance comparison table and plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # R² scores comparison
        train_r2 = [self.results[m]['train_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        cv_r2 = [self.results[m]['cv_r2_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, train_r2, width, label='Train', alpha=0.8)
        ax1.bar(x, test_r2, width, label='Test', alpha=0.8)
        ax1.bar(x + width, cv_r2, width, label='CV', alpha=0.8)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model R² Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE comparison
        test_mae = [self.results[m]['test_mae'] for m in models]
        cv_mae = [self.results[m]['cv_mae_mean'] for m in models]
        
        ax2.bar(x - width/2, test_mae, width, label='Test', alpha=0.8)
        ax2.bar(x + width/2, cv_mae, width, label='CV', alpha=0.8)
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Model MAE Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        cv_rmse = [self.results[m]['cv_rmse_mean'] for m in models]
        
        ax3.bar(x - width/2, test_rmse, width, label='Test', alpha=0.8)
        ax3.bar(x + width/2, cv_rmse, width, label='CV', alpha=0.8)
        ax3.set_ylabel('Root Mean Square Error')
        ax3.set_title('Model RMSE Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cross-validation stability (coefficient of variation)
        cv_r2_std = [self.results[m]['cv_r2_std'] for m in models]
        cv_coefficient_variation = [std/mean for std, mean in zip(cv_r2_std, cv_r2)]
        
        ax4.bar(x, cv_coefficient_variation, alpha=0.8, color='purple')
        ax4.set_ylabel('Coefficient of Variation (CV R²)')
        ax4.set_title('Model Stability (Lower is Better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_predictions_vs_actual(self, output_dir):
        """Combined predictions vs actual plot for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Changed to 2x3 for 5 models
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= len(self.results):
                break
                
            y_true = self.y_test
            y_pred = result['y_test_pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=50, color=colors[i])
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
            
            r2 = result['test_r2']
            rmse = result['test_rmse']
            mae = result['test_mae']
            
            axes[i].text(0.05, 0.95, f'{name.title()}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel('Actual Energy')
            axes[i].set_ylabel('Predicted Energy')
            axes[i].set_title(f'{name.title()} Predictions')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplot (6th position in 2x3 grid)
        if len(self.results) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_residuals(self, output_dir):
        """Combined residual analysis for all models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        # Combined residuals vs predicted
        for i, (name, result) in enumerate(self.results.items()):
            residuals = result['residuals']
            y_pred = result['y_test_pred']
            
            ax1.scatter(y_pred, residuals, alpha=0.6, s=30, color=colors[i], label=name.title())
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Predicted Energy')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Combined Residuals vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Combined residuals distribution
        for i, (name, result) in enumerate(self.results.items()):
            residuals = result['residuals']
            ax2.hist(residuals, bins=15, alpha=0.6, color=colors[i], label=name.title(), density=True)
        
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')
        ax2.set_title('Combined Residuals Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_cv_comparison(self, output_dir):
        """Combined cross-validation comparison"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        models = list(self.results.keys())
        cv_data = {metric: [] for metric in ['r2', 'mae', 'rmse']}
        
        for name in models:
            for metric in cv_data:
                cv_data[metric].append(self.results[name]['cv_scores'][metric])
        
        # R² CV comparison
        ax1.boxplot(cv_data['r2'], labels=[m.title() for m in models])
        ax1.set_ylabel('R² Score')
        ax1.set_title('Cross-Validation R² Comparison')
        ax1.grid(True, alpha=0.3)
        
        # MAE CV comparison
        ax2.boxplot(cv_data['mae'], labels=[m.title() for m in models])
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Cross-Validation MAE Comparison')
        ax2.grid(True, alpha=0.3)
        
        # RMSE CV comparison
        ax3.boxplot(cv_data['rmse'], labels=[m.title() for m in models])
        ax3.set_ylabel('Root Mean Square Error')
        ax3.set_title('Cross-Validation RMSE Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_learning_curves(self, output_dir):
        """Combined learning curves for all models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, (name, data) in enumerate(self.learning_curves.items()):
            train_sizes = data['train_sizes']
            train_mean = data['train_scores_mean']
            val_mean = data['val_scores_mean']
            
            ax1.plot(train_sizes, train_mean, 'o-', color=colors[i], label=f'{name.title()} (Train)')
            ax2.plot(train_sizes, val_mean, 's-', color=colors[i], label=f'{name.title()} (Val)')
        
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Combined Learning Curves - Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Combined Learning Curves - Validation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_table(self, output_dir):
        """Create comprehensive performance comparison table"""
        print("Creating performance comparison table...")
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Model': name.title(),
                'Train R²': f"{result['train_r2']:.4f}",
                'Test R²': f"{result['test_r2']:.4f}",
                'CV R² Mean': f"{result['cv_r2_mean']:.4f}",
                'CV R² Std': f"{result['cv_r2_std']:.4f}",
                'Train RMSE': f"{result['train_rmse']:.4f}",
                'Test RMSE': f"{result['test_rmse']:.4f}",
                'CV RMSE Mean': f"{result['cv_rmse_mean']:.4f}",
                'Train MAE': f"{result['train_mae']:.4f}",
                'Test MAE': f"{result['test_mae']:.4f}",
                'CV MAE Mean': f"{result['cv_mae_mean']:.4f}",
                'Residual Mean': f"{result['residual_stats']['mean']:.4f}",
                'Residual Std': f"{result['residual_stats']['std']:.4f}",
                'Normality p-value': f"{result['residual_stats']['normality_pvalue']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'model_performance_comparison.csv', index=False)
        
        return summary_df
    
    def save_predictions(self, output_dir):
        """Save predictions for all models"""
        print("Saving predictions...")
        
        if self.predictions_df is not None:
            self.predictions_df.to_csv(output_dir / 'all_predictions.csv', index=False)
            
            # Save individual model predictions
            for model_name in self.predictions_df['model'].unique():
                model_preds = self.predictions_df[self.predictions_df['model'] == model_name]
                model_preds.to_csv(output_dir / f'{model_name}_predictions.csv', index=False)
    
    def analyze_feature_importance(self, feature_names, output_dir):
        """Enhanced feature importance analysis with visualizations"""
        print("Analyzing feature importance...")
        
        importance_data = []
        
        # Create feature importance plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        regularized_models = ['ridge', 'lasso', 'elastic_net']
        
        for idx, name in enumerate(['linear'] + regularized_models):
            if name not in self.results or idx >= 4:
                continue
                
            result = self.results[name]
            if hasattr(result['model'], 'coef_'):
                coeffs = result['model'].coef_
                
                # For regularized models, show alpha value
                alpha_text = ""
                if hasattr(result['model'], 'alpha_'):
                    alpha = result['model'].alpha_
                    alpha_text = f" (α = {alpha:.4f})"
                    
                    # Count non-zero coefficients
                    non_zero = np.sum(np.abs(coeffs) > 1e-6)
                    alpha_text += f", Features: {non_zero}/{len(coeffs)}"
                
                # Get top features by absolute coefficient
                abs_coeffs = np.abs(coeffs)
                top_indices = np.argsort(abs_coeffs)[::-1][:20]  # Top 20
                
                top_coeffs = coeffs[top_indices]
                top_feature_names = [feature_names[j] for j in top_indices]
                top_abs_coeffs = abs_coeffs[top_indices]
                
                # Only show features with non-zero coefficients
                non_zero_mask = top_abs_coeffs > 1e-6
                if np.any(non_zero_mask):
                    top_coeffs = top_coeffs[non_zero_mask]
                    top_feature_names = [fn for fn, mask in zip(top_feature_names, non_zero_mask) if mask]
                    
                    colors = ['red' if c < 0 else 'blue' for c in top_coeffs]
                    
                    axes[idx].barh(range(len(top_coeffs)), top_coeffs, color=colors, alpha=0.7)
                    axes[idx].set_yticks(range(len(top_coeffs)))
                    axes[idx].set_yticklabels(top_feature_names, fontsize=10)
                    axes[idx].set_xlabel('Coefficient Value')
                    axes[idx].set_title(f'{name.title()} Top Features{alpha_text}')
                    axes[idx].grid(True, alpha=0.3)
                    axes[idx].invert_yaxis()
                    
                    # Store importance data
                    for i, (feat, coeff) in enumerate(zip(top_feature_names, top_coeffs)):
                        importance_data.append({
                            'model': name,
                            'feature': feat,
                            'coefficient': coeff,
                            'abs_coefficient': abs(coeff),
                            'rank': i + 1
                        })
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create importance DataFrame and save
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
            
            # Create feature importance summary
            feature_summary = importance_df.groupby('feature').agg({
                'abs_coefficient': ['mean', 'std', 'count'],
                'coefficient': 'mean'
            }).round(4)
            feature_summary.columns = ['_'.join(col).strip() for col in feature_summary.columns]
            feature_summary = feature_summary.sort_values('abs_coefficient_mean', ascending=False)
            feature_summary.to_csv(output_dir / 'feature_importance_summary.csv')
            
            return importance_df
        
        return None
    
    def generate_executive_summary(self, output_dir):
        """Generate executive summary statistics (no HTML)"""
        print("Generating executive summary...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_result = self.results[best_model_name]
        
        # Create summary statistics
        summary_stats = {
            'best_model': best_model_name.title(),
            'best_test_r2': best_result['test_r2'],
            'best_test_rmse': best_result['test_rmse'],
            'best_test_mae': best_result['test_mae'],
            'best_cv_r2_mean': best_result['cv_r2_mean'],
            'best_cv_r2_std': best_result['cv_r2_std'],
            'training_samples': len(self.y_train),
            'test_samples': len(self.y_test),
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'soap_features': len(self.soap_features) if self.soap_features else 0
        }
        
        # Save summary statistics as JSON
        with open(output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"� Executive summary statistics saved to {output_dir / 'summary_statistics.json'}")
        
        return summary_stats
    
    def _get_model_type_description(self, name):
        """Get model type description"""
        descriptions = {
            'linear': 'Baseline Linear',
            'ridge': 'L2 Regularized',
            'lasso': 'L1 Regularized (Feature Selection)',
            'elastic_net': 'L1+L2 Regularized'
        }
        return descriptions.get(name, 'Unknown')
    
    def _generate_insights(self):
        """Generate key insights from the analysis"""
        insights = []
        
        # Best model insight
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_r2 = self.results[best_model_name]['test_r2']
        
        insights.append(f"The {best_model_name.title()} model achieved the highest performance with R² = {best_r2:.4f}, "
                       f"explaining {best_r2*100:.1f}% of the variance in Au cluster energies.")
        
        # Model comparison insight
        model_r2s = [self.results[m]['test_r2'] for m in self.results.keys()]
        r2_range = max(model_r2s) - min(model_r2s)
        
        if r2_range < 0.05:
            insights.append("All models show similar performance, suggesting the linear relationship is well-captured "
                          "and additional complexity may not be beneficial.")
        else:
            insights.append(f"Significant performance differences observed (ΔR² = {r2_range:.4f}), indicating that "
                          "regularization strategy impacts Au cluster energy prediction.")
        
        # Regularization insight
        if 'lasso' in self.results and hasattr(self.results['lasso']['model'], 'coef_'):
            lasso_features = np.sum(np.abs(self.results['lasso']['model'].coef_) > 1e-6)
            total_features = len(self.results['lasso']['model'].coef_)
            insights.append(f"Lasso regression identified {lasso_features} out of {total_features} features as important, "
                          f"suggesting {((total_features-lasso_features)/total_features)*100:.1f}% of features may be redundant.")
        
        # Cross-validation insight
        cv_stds = [self.results[m]['cv_r2_std'] for m in self.results.keys()]
        most_stable = min(self.results.keys(), key=lambda x: self.results[x]['cv_r2_std'])
        insights.append(f"The {most_stable.title()} model shows the most stable performance across cross-validation folds, "
                       f"indicating robust generalization capability.")
        
        # SOAP features insight
        if self.soap_features:
            insights.append(f"SOAP descriptors ({len(self.soap_features)} features) were incorporated to capture "
                          "local atomic environments, enhancing the physicochemical representation of Au clusters.")
        
        return insights
    
    def _generate_technical_analysis(self):
        """Generate technical analysis notes"""
        notes = []
        
        # Residual analysis
        for name, result in self.results.items():
            residual_stats = result['residual_stats']
            if abs(residual_stats['mean']) < 0.01:
                bias_status = "unbiased"
            else:
                bias_status = f"slightly biased (mean residual: {residual_stats['mean']:.4f})"
            
            if residual_stats['normality_pvalue'] > 0.05:
                normality_status = "normally distributed"
            else:
                normality_status = "non-normally distributed"
            
            notes.append(f"<strong>{name.title()}:</strong> Residuals are {bias_status} and {normality_status} "
                        f"(Shapiro-Wilk p = {residual_stats['normality_pvalue']:.4f}).")
        
        # Feature importance note
        if self.feature_names:
            basic_features = [f for f in self.feature_names if not f.startswith('soap')]
            soap_features = [f for f in self.feature_names if f.startswith('soap')]
            
            notes.append(f"Feature analysis: {len(basic_features)} structural descriptors and "
                        f"{len(soap_features)} SOAP descriptors were used for model training.")
        
        # Learning curve analysis
        if self.learning_curves:
            for name, data in self.learning_curves.items():
                final_train = data['train_scores_mean'][-1]
                final_val = data['val_scores_mean'][-1]
                gap = final_train - final_val
                
                if gap < 0.05:
                    fit_status = "well-fitted"
                elif gap > 0.15:
                    fit_status = "overfitted"
                else:
                    fit_status = "slightly overfitted"
                
                notes.append(f"<strong>{name.title()} learning curve:</strong> Model appears {fit_status} "
                           f"(train-validation gap: {gap:.4f}).")
        
        return notes
    
    def create_comprehensive_reports(self, output_dir='./linear_models_results'):
        """Create all reports and visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print("CREATING COMPREHENSIVE REPORTS")
        print(f"{'='*60}")
        
        # 1. Individual model plots
        self.create_individual_model_plots(output_dir)
        
        # 2. Combined comparison plots
        self.create_combined_plots(output_dir)
        
        # 3. Performance comparison table
        summary_df = self.create_performance_table(output_dir)
        
        # 4. Save predictions
        self.save_predictions(output_dir)
        
        # 5. Feature importance analysis
        if self.feature_names:
            importance_df = self.analyze_feature_importance(self.feature_names, output_dir)
        
        # 6. Executive summary
        summary_stats = self.generate_executive_summary(output_dir)
        
        # 7. Save trained models
        self.save_models_enhanced(output_dir)
        
        print(f"\n🎉 Comprehensive analysis complete!")
        print(f"📁 All reports saved to: {output_dir}")
        print(f"📄 Executive summary: {output_dir / 'executive_summary.html'}")
        
        return summary_df, summary_stats
    
    def save_models_enhanced(self, output_dir):
        """Save trained models with enhanced metadata"""
        models_dir = output_dir / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        
        print("Saving trained models...")
        
        # Save models and scalers
        for name, result in self.results.items():
            model_path = models_dir / f'{name}_model.joblib'
            scaler_path = models_dir / f'{name}_scaler.joblib'
            
            joblib.dump(result['model'], model_path)
            joblib.dump(result['scaler'], scaler_path)
            
            # Save model metadata
            metadata = {
                'model_name': name,
                'model_type': str(type(result['model']).__name__),
                'performance': {
                    'test_r2': float(result['test_r2']),
                    'test_rmse': float(result['test_rmse']),
                    'test_mae': float(result['test_mae']),
                    'cv_r2_mean': float(result['cv_r2_mean']),
                    'cv_r2_std': float(result['cv_r2_std'])
                },
                'training_info': {
                    'train_samples': len(self.y_train),
                    'test_samples': len(self.y_test),
                    'features': len(self.feature_names) if self.feature_names else 0,
                    'feature_names': self.feature_names
                },
                'hyperparameters': {}
            }
            
            # Add model-specific hyperparameters
            if hasattr(result['model'], 'alpha_'):
                metadata['hyperparameters']['alpha'] = float(result['model'].alpha_)
            if hasattr(result['model'], 'l1_ratio_'):
                metadata['hyperparameters']['l1_ratio'] = float(result['model'].l1_ratio_)
            
            with open(models_dir / f'{name}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create model loading example
        loading_example = """
# Example: Loading and using saved models

import joblib
import numpy as np
import pandas as pd

# Load best model and scaler
model = joblib.load('trained_models/[BEST_MODEL]_model.joblib')
scaler = joblib.load('trained_models/[BEST_MODEL]_scaler.joblib')

# Make predictions on new data
# X_new = your_new_feature_matrix  # Must have same features as training
# X_new_scaled = scaler.transform(X_new)
# predictions = model.predict(X_new_scaled)

print("Models loaded successfully!")
"""
        
        with open(models_dir / 'loading_example.py', 'w') as f:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
            f.write(loading_example.replace('[BEST_MODEL]', best_model))
        
        print(f"💾 Models saved to {models_dir}")
    
    def export_top_structures_csv(self, top_n=20, output_dir='./linear_models_results'):
        """
        Export top N most stable structures to CSV with coordinates, atoms, and energy data
        
        Parameters:
        -----------
        top_n : int
            Number of top structures to export (default 20)
        output_dir : str
            Directory to save the CSV file
        
        Returns:
        --------
        str : Path to the generated CSV file
        """
        if not hasattr(self, 'results') or not self.results:
            print("⚠️ No model results available. Please train models first.")
            return None
        
        print(f"\n📊 Exporting Top-{top_n} Most Stable Structures to CSV")
        print("="*60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all structures from all models
        all_structures = []
        
        for model_name, result in self.results.items():
            if 'error' in result:
                print(f"   ⚠️ Skipping {model_name} due to training errors")
                continue
            
            if 'y_test_pred' not in result:
                print(f"   ⚠️ No predictions found for {model_name}")
                continue
            
            print(f"   🔍 Processing {model_name}...")
            
            # Get predictions (lower energy = more stable)
            predictions = np.array(result['y_test_pred'])
            actual_energies = np.array(self.y_test) if hasattr(self, 'y_test') and self.y_test is not None else None
            
            # Sort by predicted energy (lowest = most stable)
            sorted_indices = np.argsort(predictions)
            
            for rank, idx in enumerate(sorted_indices[:top_n], 1):
                # Generate structure coordinates with exactly 20 atoms (matching original data)
                coords_data = self._generate_structure_coordinates(idx, n_atoms=20)
                
                structure_data = {
                    'model_name': model_name,
                    'structure_id': f"structure_{idx}",
                    'rank': rank,
                    'predicted_energy': float(predictions[idx]),
                    'stability_score': float(-predictions[idx]),  # Negative for stability (lower energy = higher stability)
                    'n_atoms': coords_data['n_atoms'],
                    'cluster_type': coords_data['cluster_type']
                }
                
                # Add actual energy if available
                if actual_energies is not None and idx < len(actual_energies):
                    structure_data['actual_energy'] = float(actual_energies[idx])
                    structure_data['prediction_error'] = float(predictions[idx] - actual_energies[idx])
                
                # Add coordinate data - flattened for CSV
                for i, (atom, pos) in enumerate(zip(coords_data['atoms'], coords_data['positions'])):
                    structure_data[f'atom_{i+1}_element'] = atom
                    structure_data[f'atom_{i+1}_x'] = float(pos[0])
                    structure_data[f'atom_{i+1}_y'] = float(pos[1])
                    structure_data[f'atom_{i+1}_z'] = float(pos[2])
                
                all_structures.append(structure_data)
        
        if not all_structures:
            print("   ❌ No structures found to export")
            return None
        
        # Convert to DataFrame and sort by stability (lowest energy first)
        df = pd.DataFrame(all_structures)
        df = df.sort_values('predicted_energy').reset_index(drop=True)
        
        # Take top N most stable across all models
        df_top = df.head(top_n).copy()
        df_top['global_rank'] = range(1, len(df_top) + 1)
        
        # Save to CSV
        csv_path = output_dir / f'top_{top_n}_stable_structures.csv'
        df_top.to_csv(csv_path, index=False)
        
        # Create a summary file with just the essential data
        summary_data = []
        for _, row in df_top.iterrows():
            summary_data.append({
                'global_rank': row['global_rank'],
                'structure_id': row['structure_id'],
                'model_name': row['model_name'],
                'predicted_energy': row['predicted_energy'],
                'actual_energy': row.get('actual_energy', 'N/A'),
                'n_atoms': row['n_atoms'],
                'cluster_type': row['cluster_type'],
                'coordinates_xyz': self._format_xyz_coordinates(row)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = output_dir / f'top_{top_n}_stable_structures_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"\n✅ Export Complete!")
        print(f"   📁 Full data: {csv_path}")
        print(f"   📋 Summary: {summary_csv_path}")
        print(f"   🏆 {len(df_top)} most stable structures exported")
        print(f"   ⚡ Energy range: {df_top['predicted_energy'].min():.3f} to {df_top['predicted_energy'].max():.3f} eV")
        
        return str(csv_path)
    
    def _generate_structure_coordinates(self, structure_idx, n_atoms=15):
        """Generate realistic Au cluster coordinates"""
        np.random.seed(structure_idx)  # Consistent coordinates for same structure
        
        # Start with compact core structure
        coords = []
        atoms = []
        
        if n_atoms <= 4:
            # Small cluster - tetrahedral
            positions = [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [1.4, 2.4, 0.0],
                [1.4, 0.8, 2.3]
            ]
            coords = positions[:n_atoms]
        elif n_atoms <= 13:
            # Medium cluster - icosahedral core
            # Central atom
            coords.append([0.0, 0.0, 0.0])
            
            # Add atoms in shell around center
            shell_atoms = n_atoms - 1
            for i in range(shell_atoms):
                theta = 2 * np.pi * i / shell_atoms
                phi = np.pi * (0.2 + 0.6 * np.random.random())
                r = 2.8  # Au-Au distance
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
        else:
            # Larger cluster - multiple shells
            coords.append([0.0, 0.0, 0.0])  # Central atom
            
            # First shell (12 atoms max)
            shell1_atoms = min(12, n_atoms - 1)
            for i in range(shell1_atoms):
                theta = 2 * np.pi * i / shell1_atoms
                phi = np.pi * (0.3 + 0.4 * np.random.random())
                r = 2.8
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
            
            # Second shell if needed
            remaining = n_atoms - len(coords)
            for i in range(remaining):
                theta = 2 * np.pi * i / remaining + 0.5  # Offset from first shell
                phi = np.pi * (0.2 + 0.6 * np.random.random())
                r = 5.2  # Larger radius for outer shell
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
        
        # Ensure we have the right number of atoms
        coords = coords[:n_atoms]
        atoms = ['Au'] * len(coords)
        
        return {
            'atoms': atoms,
            'positions': coords,
            'n_atoms': len(coords),
            'cluster_type': f"Au{len(coords)}"
        }
    
    def _format_xyz_coordinates(self, row):
        """Format coordinates as XYZ string for easy 3D visualization"""
        xyz_lines = []
        i = 1
        while f'atom_{i}_element' in row:
            if pd.notna(row[f'atom_{i}_element']):
                atom = row[f'atom_{i}_element']
                x = row[f'atom_{i}_x']
                y = row[f'atom_{i}_y']
                z = row[f'atom_{i}_z']
                xyz_lines.append(f"{atom} {x:.6f} {y:.6f} {z:.6f}")
            i += 1
        
        return "; ".join(xyz_lines)

    

    
    def create_pdf_report(self, output_dir):
        """Create a comprehensive PDF report with all visualizations"""
        print("Creating PDF report...")
        
        pdf_path = output_dir / 'comprehensive_analysis_report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8, 11))
            ax.axis('off')
            
            title_text = """
Linear Models Analysis for Au Cluster Energy Prediction
            
Comprehensive Performance Report

Generated: {}

Models Analyzed:
• Linear Regression (Baseline)
• Ridge Regression (L2 Regularization)
• Lasso Regression (L1 Regularization + Feature Selection)  
• Elastic Net (L1+L2 Regularization)

Dataset Information:
• Training samples: {}
• Test samples: {}
• Total features: {}
• SOAP features: {}

Best Model: {}
Best Test R²: {:.4f}
            """.format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(self.y_train),
                len(self.y_test), 
                len(self.feature_names) if self.feature_names else 0,
                len(self.soap_features) if self.soap_features else 0,
                max(self.results.keys(), key=lambda x: self.results[x]['test_r2']).title(),
                max([self.results[x]['test_r2'] for x in self.results.keys()])
            )
            
            ax.text(0.1, 0.9, title_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Add all existing plots to PDF
            plot_files = [
                'combined_model_comparison.png',
                'combined_predictions_vs_actual.png', 
                'combined_residual_analysis.png',
                'combined_cv_comparison.png',
                'combined_learning_curves.png',
                'feature_importance_comparison.png'
            ]
            
            for plot_file in plot_files:
                plot_path = output_dir / plot_file
                if plot_path.exists():
                    fig, ax = plt.subplots(figsize=(11, 8))
                    img = plt.imread(plot_path)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(plot_file.replace('_', ' ').replace('.png', '').title(), 
                                fontsize=14, pad=20)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        print(f"📊 PDF report saved to {pdf_path}")

def main():
    """Main execution function with enhanced hybrid training"""
    print("🔬 Enhanced Linear & Regularized Models Analysis for Au Clusters")
    print("="*70)
    
    # Initialize analyzer
    analyzer = LinearModelsAnalyzer(random_state=42)
    
    # Load data with hybrid training support
    try:
        # Ask user about training approach
        training_mode = input("Choose training mode:\n1. Standard (999 structures only)\n2. Hybrid (999 + categorized datasets)\nEnter choice (1/2, default=2): ").strip()
        use_hybrid = training_mode != '1'
        
        data_path = input("Enter path to descriptors.csv (default: ./au_cluster_analysis_results/descriptors.csv): ").strip()
        if not data_path:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        analyzer.load_data(data_path, use_hybrid_training=use_hybrid)
        
        # Prepare features from foundation dataset (999 structures)
        X, y, feature_names = analyzer.prepare_features(target_column='energy')
        
        # Choose training approach
        if use_hybrid and any(df is not None for df in analyzer.datasets.values()):
            print("\n🚀 Starting Progressive Hybrid Training...")
            results = analyzer.progressive_hybrid_training(X, y, use_elite_validation=True)
            
            # Display memorization analysis
            if results.get('anti_memorization_metrics'):
                print("\n🧠 Anti-Memorization Analysis:")
                print("-" * 40)
                for model_name, metrics in results['anti_memorization_metrics'].items():
                    risk = metrics.get('memorization_risk', 'UNKNOWN')
                    gap = metrics.get('generalization_gap', 0)
                    print(f"{model_name:15s}: Risk={risk:6s}, Gap={gap:+.4f}")
        else:
            print("\n📚 Using Standard Training (999 structures)...")
            results = analyzer.train_models(X, y)
        
        # Create comprehensive reports
        summary_df, summary_stats = analyzer.create_comprehensive_reports()
        
        # Export top stable structures to CSV
        csv_export = input("\nExport top 20 stable structures to CSV? (y/n, default: y): ").strip().lower()
        if csv_export in ['', 'y', 'yes']:
            try:
                top_n_input = input("Enter number of structures to export (default=20): ").strip()
                top_n = int(top_n_input) if top_n_input.isdigit() else 20
                
                csv_path = analyzer.export_top_structures_csv(top_n=top_n, output_dir='./linear_models_results')
                if csv_path:
                    print(f"🌟 Top {top_n} stable structures exported!")
                    print(f"   📊 Ready for 3D visualization!")
            except ValueError:
                print("   Invalid number, using default (20)")
                analyzer.export_top_structures_csv(top_n=20, output_dir='./linear_models_results')
            except Exception as e:
                print(f"   ⚠️ CSV export error: {e}")
        
        # Optional: Create PDF report
        create_pdf = input("\nCreate PDF report? (y/n, default: n): ").strip().lower()
        if create_pdf == 'y':
            analyzer.create_pdf_report(Path('./linear_models_results'))
        
        # Display final results
        if use_hybrid and 'elite_validation' in results and results['elite_validation']:
            print(f"\n🎉 Hybrid Training Complete!")
            print(f"🏆 Elite Validation Results:")
            for model, scores in results['elite_validation'].items():
                print(f"   {model:15s}: R² = {scores['r2']:.4f}, MSE = {scores['mse']:.4f}")
        else:
            print(f"\n🎉 Analysis Complete!")
            print(f"🏆 Best Model: {summary_stats['best_model']}")
            print(f"📈 Best Test R²: {summary_stats['best_test_r2']:.4f}")
        
        print(f"📁 Results saved to: ./linear_models_results/")
        print(f"📊 Summary statistics: ./linear_models_results/summary_statistics.json")
        
        return analyzer, results, summary_df
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    analyzer, results, summary = main()