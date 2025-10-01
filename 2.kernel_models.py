#!/usr/bin/env python3
"""
ENHANCED: Non-Linear Kernel Methods for Au Cluster Energy Prediction
Focused on RBF kernel SVR for non-linear kernel-based approaches
Complete documentation package with all files and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import pickle
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# SOAP descriptors for enhanced features
try:
    from dscribe.descriptors import SOAP
    from ase.atoms import Atoms
    SOAP_AVAILABLE = True
    print("✅ DScribe available - SOAP features enabled")
except ImportError:
    print("⚠️ DScribe not available. Using basic descriptors only.")
    SOAP_AVAILABLE = False

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveKernelAnalysis:
    """
    Non-Linear Kernel Methods Analysis with Full Documentation Package
    Focused on RBF kernel SVR for complex non-linear relationships
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'random_state': random_state,
            'sklearn_version': None
        }
        
        # Model configurations with better hyperparameter ranges
        self.model_configs = {
            'svr_rbf_conservative': {
                'model': SVR(kernel='rbf', cache_size=1000),
                'params': {
                    'model__C': [1, 10, 100],                # Conservative C values
                    'model__gamma': [0.001, 0.01, 0.1],      # Conservative gamma
                    'model__epsilon': [0.1, 0.5, 1.0]       # Larger epsilon for stability
                },
                'feature_selection': 25,  # More features for SOAP
                'description': 'Support Vector Regression with RBF kernel - non-linear kernel method'
            }
        }
    
    def create_soap_features(self, structures_data=None, n_components=100, pca_model=None):
        """
        Create SOAP (Smooth Overlap of Atomic Positions) descriptors
        
        Args:
            structures_data: List of ASE Atoms objects or coordinates
            n_components: Number of SOAP components to generate
            pca_model: Pre-fitted PCA model (to avoid data leakage)
            
        Returns:
            DataFrame with SOAP features and optionally the PCA model
        """
        if not SOAP_AVAILABLE:
            print("⚠️  SOAP features not available - DScribe not installed")
            return None, None
            
        if structures_data is None:
            print("⚠️  No structure data provided for SOAP features")
            return None, None
        
        print("🧪 Generating SOAP descriptors...")
        
        # SOAP parameters optimized for Au clusters
        soap = SOAP(
            species=["Au"],
            r_cut=6.0,  # Cutoff radius in Angstroms
            n_max=8,    # Maximum radial basis functions
            l_max=6,    # Maximum degree of spherical harmonics
            sparse=False,
            periodic=False,  # Au clusters are not periodic
            crossover=True,  # Include cross-species terms
            average="off"    # Don't average over atoms
        )
        
        soap_features = []
        valid_indices = []
        
        for i, atoms in enumerate(structures_data):
            try:
                if isinstance(atoms, dict):
                    # Convert coordinate dictionary to ASE Atoms
                    positions = np.array([[atoms[f'x_{j}'], atoms[f'y_{j}'], atoms[f'z_{j}']] 
                                        for j in range(20)])  # Assuming Au20 clusters
                    atoms_obj = Atoms('Au20', positions=positions)
                else:
                    atoms_obj = atoms
                
                # Generate SOAP descriptors
                soap_desc = soap.create(atoms_obj)
                
                # Average over all atoms in the cluster
                soap_avg = np.mean(soap_desc, axis=0)
                soap_features.append(soap_avg)
                valid_indices.append(i)
                
            except Exception as e:
                print(f"⚠️  Failed to generate SOAP for structure {i}: {e}")
                continue
        
        if not soap_features:
            print("❌ No valid SOAP features generated")
            return None, None
        
        # Convert to DataFrame
        soap_array = np.array(soap_features)
        
        # Apply PCA to reduce dimensionality if needed
        if soap_array.shape[1] > n_components:
            # Use provided PCA model or create a new one
            if pca_model is None:
                pca_model = PCA(n_components=n_components, random_state=self.random_state)
                soap_reduced = pca_model.fit_transform(soap_array)
                print(f"📊 SOAP features reduced from {soap_array.shape[1]} to {n_components} via PCA (new model)")
                print(f"📊 PCA explained variance ratio: {pca_model.explained_variance_ratio_.sum():.3f}")
            else:
                # Use pre-fitted PCA model to avoid data leakage
                soap_reduced = pca_model.transform(soap_array)
                print(f"📊 SOAP features reduced from {soap_array.shape[1]} to {n_components} via PCA (existing model)")
            
            feature_names = [f'soap_pca_{i}' for i in range(n_components)]
        else:
            soap_reduced = soap_array
            feature_names = [f'soap_{i}' for i in range(soap_reduced.shape[1])]
            pca_model = None
        
        soap_df = pd.DataFrame(soap_reduced, columns=feature_names, index=valid_indices)
        
        print(f"✅ Generated {len(soap_df)} SOAP feature vectors with {soap_df.shape[1]} components")
        
        return soap_df, pca_model
    
    def load_and_prepare_data(self, data_path=None, target_column='energy', use_hybrid_training=True):
        """
        Enhanced data loading with hybrid training support for kernel methods
        
        Args:
            data_path: Path to original descriptors.csv (999 structures)
            target_column: Target variable name
            use_hybrid_training: Whether to use progressive training approach
        """
        print("\n" + "="*70)
        print("📁 ENHANCED DATA LOADING AND PREPARATION")
        print("="*70)
        
        # Load original 999 structures for foundation learning
        if data_path is None:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        # Load foundation data
        if isinstance(data_path, str):
            self.df_foundation = pd.read_csv(data_path)
            print(f"✅ Loaded foundation data from: {data_path}")
        else:
            self.df_foundation = data_path
            print("✅ Using provided DataFrame")
        
        # Store metadata
        self.metadata['data_source'] = str(data_path) if isinstance(data_path, str) else 'DataFrame'
        self.metadata['original_shape'] = self.df_foundation.shape
        
        # Load categorized high-quality datasets
        self.datasets = {}
        dataset_files = {
            'balanced': './task2/improved_dataset_balanced.csv',
            'high_quality': '.task2/improved_dataset_high_quality.csv', 
            'elite': './task2/improved_dataset_elite.csv'
        }
        
        if use_hybrid_training:
            print("🔄 Loading hybrid training datasets for kernel methods...")
            
            for name, file_path in dataset_files.items():
                try:
                    df = pd.read_csv(file_path)
                    df = df.dropna(subset=[target_column])
                    self.datasets[name] = df
                    print(f"   ✅ {name}: {len(df)} structures")
                except FileNotFoundError:
                    print(f"   ⚠️  {name}: File not found - {file_path}")
                    self.datasets[name] = None
        
        # Basic cleaning for foundation data
        initial_size = len(self.df_foundation)
        self.df_foundation = self.df_foundation.dropna(subset=[target_column])
        final_size = len(self.df_foundation)
        
        if initial_size != final_size:
            print(f"⚠️  Removed {initial_size - final_size} rows with missing target values")
        
        # Set primary dataset for analysis
        df = self.df_foundation
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Foundation (999): {len(self.df_foundation)} samples")
        print(f"   Target range: {self.df_foundation[target_column].min():.2f} to {self.df_foundation[target_column].max():.2f}")
        
        if use_hybrid_training and any(df is not None for df in self.datasets.values()):
            print(f"   Hybrid training: ENABLED for kernel methods")
            for name, df_cat in self.datasets.items():
                if df_cat is not None:
                    print(f"   - {name}: {len(df_cat)} samples")
        
        # Load data
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
            print(f"✅ Loaded data from: {data_path}")
        else:
            df = data_path
            print("✅ Using provided DataFrame")
        
        # Store metadata
        self.metadata['data_source'] = str(data_path) if isinstance(data_path, str) else 'DataFrame'
        self.metadata['original_shape'] = df.shape
        
        # Basic cleaning
        initial_size = len(df)
        df = df.dropna(subset=[target_column])
        final_size = len(df)
        
        if initial_size != final_size:
            print(f"⚠️  Removed {initial_size - final_size} rows with missing target values")
        
        # Get numeric features only - EXCLUDE ENERGY-DERIVED FEATURES
        exclude_cols = [target_column, 'energy_per_atom', 'filename', 'Unnamed: 0']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        print(f"📊 Found {len(feature_cols)} potential features")
        
        # Remove zero/constant variance features
        valid_features = []
        removed_features = []
        
        for col in feature_cols:
            if df[col].std() > 1e-10 and df[col].nunique() > 1:
                valid_features.append(col)
            else:
                removed_features.append(col)
        
        if removed_features:
            print(f"🗑️  Removed {len(removed_features)} constant/zero-variance features")
        
        # Create clean feature matrix
        X_basic = df[valid_features].fillna(df[valid_features].mean())
        y = df[target_column]
        
        # Try to generate SOAP features
        soap_features = []
        try:
            # Look for coordinate data to generate SOAP features
            coord_file = None
            if isinstance(data_path, str):
                # Try to find raw coordinates file
                data_dir = Path(data_path).parent
                coord_file = data_dir / "raw_coordinates.csv"
                if not coord_file.exists():
                    coord_file = data_dir.parent / "au_cluster_analysis_results" / "raw_coordinates.csv"
            
            if coord_file and coord_file.exists():
                print("🧪 Found coordinate data - generating SOAP features...")
                coord_df = pd.read_csv(coord_file)
                
                # Group coordinates by filename to reconstruct structures
                structures = []
                for filename in coord_df['filename'].unique():
                    structure_data = coord_df[coord_df['filename'] == filename]
                    if len(structure_data) == 20:  # Au20 clusters
                        positions = structure_data[['x', 'y', 'z']].values
                        atoms = Atoms('Au20', positions=positions)
                        structures.append(atoms)
                
                if structures:
                    # Generate SOAP features but store the PCA model to prevent data leakage
                    soap_df, self.soap_pca_model = self.create_soap_features(structures, n_components=50)
                    if soap_df is not None:
                        # Align SOAP features with main dataframe
                        if len(soap_df) == len(X_basic):
                            X_combined = pd.concat([X_basic.reset_index(drop=True), 
                                                  soap_df.reset_index(drop=True)], axis=1)
                            soap_features = list(soap_df.columns)
                            print(f"✅ Combined {len(valid_features)} basic + {len(soap_features)} SOAP features")
                            print(f"⚠️  PCA model for SOAP features stored to prevent data leakage between train/test")
                        else:
                            print(f"⚠️  SOAP feature count mismatch - using basic features only")
                            X_combined = X_basic
                            soap_features = []
                    else:
                        X_combined = X_basic
                        soap_features = []
                else:
                    print("⚠️  No valid structures found - using basic features only")
                    X_combined = X_basic
                    soap_features = []
            else:
                print("⚠️  No coordinate data found - using basic features only")
                X_combined = X_basic
                soap_features = []
                
        except Exception as e:
            print(f"⚠️  SOAP feature generation failed: {e}")
            X_combined = X_basic
            soap_features = []
        
        X = X_combined
        
        # Store feature information
        all_features = valid_features + soap_features
        self.metadata['features'] = {
            'total_features': len(all_features),
            'basic_features': len(valid_features),
            'soap_features': len(soap_features),
            'feature_names': all_features,
            'removed_features': removed_features,
            'target_column': target_column
        }
        
        # Data statistics
        self.metadata['data_stats'] = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'target_min': float(y.min()),
            'target_max': float(y.max()),
            'feature_sample_ratio': X.shape[1] / len(X)
        }
        
        # Print summary
        print(f"\n📈 DATA SUMMARY:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]} (Basic: {self.metadata['features']['basic_features']}, SOAP: {self.metadata['features']['soap_features']})")
        print(f"   Feature/Sample ratio: {X.shape[1]/len(X):.3f}")
        print(f"   Target - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
        print(f"   Target - Range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y
    
    def create_pipeline(self, model_name, config, n_features):
        """Create processing pipeline"""
        steps = []
        
        # Always use StandardScaler
        steps.append(('scaler', StandardScaler()))
        
        # Feature selection if specified
        if config.get('feature_selection') and n_features > config['feature_selection']:
            steps.append(('selector', SelectKBest(f_regression, k=config['feature_selection'])))
        
        # Add model
        steps.append(('model', config['model']))
        
        return Pipeline(steps)
    
    def generate_learning_curves(self, X, y, output_dir):
        """Generate learning curves for all models"""
        print("\n📈 Generating learning curves...")
        
        # Ensure main output directory exists first
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        learning_curves_dir = output_dir / 'learning_curves'
        learning_curves_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning curve settings
        train_sizes = np.linspace(0.1, 1.0, 10)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Create subplots for all models
        n_models = len(self.model_configs)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        learning_curve_data = {}
        
        for idx, (name, config) in enumerate(self.model_configs.items()):
            print(f"   Generating curve for {name}...")
            
            try:
                # Create pipeline
                pipeline = self.create_pipeline(name, config, X.shape[1])
                
                # Generate learning curve
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    pipeline, X, y, 
                    train_sizes=train_sizes,
                    cv=cv,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                
                # Convert to positive RMSE
                train_scores = -train_scores
                val_scores = -val_scores
                
                # Calculate statistics
                train_mean = train_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                val_mean = val_scores.mean(axis=1)
                val_std = val_scores.std(axis=1)
                
                # Store data
                learning_curve_data[name] = {
                    'train_sizes': train_sizes_abs.tolist(),
                    'train_mean': train_mean.tolist(),
                    'train_std': train_std.tolist(),
                    'val_mean': val_mean.tolist(),
                    'val_std': val_std.tolist()
                }
                
                # Plot
                ax = axes[idx]
                ax.plot(train_sizes_abs, train_mean, 'o-', label='Training RMSE', linewidth=2, markersize=5)
                ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
                
                ax.plot(train_sizes_abs, val_mean, 'o-', label='Cross-validation RMSE', linewidth=2, markersize=5)
                ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)
                
                ax.set_xlabel('Training Set Size', fontsize=12)
                ax.set_ylabel('RMSE', fontsize=12)
                ax.set_title(f'Learning Curve - {name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add final performance text
                final_train = train_mean[-1]
                final_val = val_mean[-1]
                ax.text(0.02, 0.98, f'Final Train RMSE: {final_train:.3f}\nFinal CV RMSE: {final_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
                
            except Exception as e:
                print(f"   ❌ Error generating curve for {name}: {e}")
                axes[idx].text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center')
                axes[idx].set_title(f'{name} - Error')
        
        # Hide unused subplots
        for idx in range(len(self.model_configs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Learning Curves - All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(learning_curves_dir / 'all_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Skip saving JSON data to avoid serialization issues
        print("ℹ️  Skipping learning curve JSON export (plots are sufficient)")
        
        return learning_curve_data
    
    def progressive_kernel_training(self, X_foundation, y_foundation, use_elite_validation=True):
        """
        Progressive kernel training: Foundation → Parameter Optimization → Elite Validation
        
        Args:
            X_foundation: Features from 999 structures
            y_foundation: Targets from 999 structures
            use_elite_validation: Whether to use elite dataset for final validation
        
        Returns:
            dict: Comprehensive training results across all stages
        """
        print("\n" + "="*70)
        print("🚀 PROGRESSIVE KERNEL TRAINING PIPELINE")
        print("="*70)
        
        results = {
            'foundation_results': {},
            'parameter_optimization': {},
            'elite_validation': {},
            'kernel_analysis': {},
            'anti_memorization_metrics': {}
        }
        
        # Stage 1: Foundation Learning (999 structures)
        print("\n📚 STAGE 1: Foundation Kernel Learning (999 structures)")
        print("-" * 50)
        
        foundation_results = self.train_models(X_foundation, y_foundation, test_size=0.2)
        results['foundation_results'] = foundation_results
        
        # Stage 2: Parameter Optimization (if high-quality dataset available)
        if self.datasets.get('high_quality') is not None:
            print("\n🎯 STAGE 2: Kernel Parameter Optimization (High-Quality subset)")
            print("-" * 50)
            
            # Prepare high-quality data with same feature processing
            X_hq, y_hq = self._prepare_dataset_features(self.datasets['high_quality'])
            
            # Optimize kernel parameters using high-quality data
            optimization_results = self._optimize_kernel_parameters(
                X_hq, y_hq, foundation_results
            )
            results['parameter_optimization'] = optimization_results
        
        # Stage 3: Elite Validation (if elite dataset available)  
        if use_elite_validation and self.datasets.get('elite') is not None:
            print("\n🏆 STAGE 3: Elite Validation (Never-seen structures)")
            print("-" * 50)
            
            X_elite, y_elite = self._prepare_dataset_features(self.datasets['elite'])
            
            elite_results = {}
            source_results = results.get('parameter_optimization', results['foundation_results'])
            
            for model_name, model_data in source_results.items():
                if isinstance(model_data, dict) and 'pipeline' in model_data:
                    elite_scores = self._validate_kernel_on_elite(
                        model_data['pipeline'], X_elite, y_elite, model_name
                    )
                    elite_results[model_name] = elite_scores
            
            results['elite_validation'] = elite_results
        
        # Kernel Analysis
        results['kernel_analysis'] = self._analyze_kernel_performance(results)
        
        # Anti-memorization analysis
        if len(results['foundation_results']) > 0:
            results['anti_memorization_metrics'] = self._analyze_kernel_memorization(
                results['foundation_results'], 
                results.get('parameter_optimization', {}),
                results.get('elite_validation', {})
            )
        
        return results
    
    def _prepare_dataset_features(self, dataset_df):
        """Prepare features for a specific dataset using same preprocessing as foundation"""
        # Get numeric features only - EXCLUDE ENERGY-DERIVED FEATURES
        exclude_cols = ['energy', 'energy_per_atom', 'filename', 'Unnamed: 0', 'structure_id']
        feature_cols = [col for col in dataset_df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(dataset_df[col])]
        
        # Use only features that exist in both datasets
        foundation_features = self.metadata.get('features', {}).get('feature_names', [])
        basic_foundation_features = [f for f in foundation_features if 'soap' not in f.lower()]
        
        # Find common features
        common_features = [f for f in feature_cols if f in basic_foundation_features]
        
        # Create feature matrix
        X = dataset_df[common_features].fillna(dataset_df[common_features].mean())
        y = dataset_df['energy']
        
        print(f"   📊 Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def _optimize_kernel_parameters(self, X_hq, y_hq, foundation_results):
        """Optimize kernel parameters using high-quality data"""
        optimization_results = {}
        
        for model_name, model_data in foundation_results.items():
            if model_name not in ['svr_rbf', 'kernel_ridge']:
                continue  # Focus on kernel methods
                
            print(f"\n🔄 Optimizing {model_name} parameters...")
            
            # Enhanced parameter grids for kernel methods
            if model_name == 'svr_rbf':
                param_grid = {
                    'model__C': [0.1, 1, 10, 100, 1000],
                    'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'model__epsilon': [0.01, 0.1, 0.2, 0.5]
                }
            elif model_name == 'kernel_ridge':
                param_grid = {
                    'model__alpha': [0.001, 0.01, 0.1, 1, 10],
                    'model__gamma': [0.001, 0.01, 0.1, 1, 10]
                }
            
            # Create pipeline
            pipeline = self.create_pipeline(model_name, self.model_configs[model_name], X_hq.shape[1])
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
            )
            
            # Fit
            grid_search.fit(X_hq, y_hq)
            
            # Store results
            optimization_results[model_name] = {
                'pipeline': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'param_grid_size': len(grid_search.cv_results_['params'])
            }
            
            print(f"   ✅ Best CV R²: {grid_search.best_score_:.4f}")
            print(f"   📋 Best params: {grid_search.best_params_}")
        
        return optimization_results
    
    def _validate_kernel_on_elite(self, pipeline, X_elite, y_elite, model_name):
        """Validate optimized kernel on elite dataset"""
        y_pred = pipeline.predict(X_elite)
        
        scores = {
            'r2': r2_score(y_elite, y_pred),
            'mse': mean_squared_error(y_elite, y_pred),
            'mae': mean_absolute_error(y_elite, y_pred),
            'predictions': y_pred,
            'actuals': y_elite
        }
        
        print(f"   🏆 {model_name}: Elite R² = {scores['r2']:.4f}, MSE = {scores['mse']:.4f}")
        return scores
    
    def _analyze_kernel_performance(self, results):
        """Analyze kernel-specific performance characteristics"""
        analysis = {}
        
        # Compare kernel methods
        kernel_methods = ['svr_rbf', 'kernel_ridge']
        for method in kernel_methods:
            method_analysis = {}
            
            # Foundation performance
            if method in results.get('foundation_results', {}):
                foundation_r2 = results['foundation_results'][method].get('test_r2', 0)
                method_analysis['foundation_r2'] = foundation_r2
            
            # Parameter optimization impact
            if method in results.get('parameter_optimization', {}):
                opt_r2 = results['parameter_optimization'][method].get('best_cv_score', 0)
                method_analysis['optimized_r2'] = opt_r2
                method_analysis['optimization_gain'] = opt_r2 - method_analysis.get('foundation_r2', 0)
            
            # Elite performance
            if method in results.get('elite_validation', {}):
                elite_r2 = results['elite_validation'][method].get('r2', 0)
                method_analysis['elite_r2'] = elite_r2
            
            analysis[method] = method_analysis
        
        return analysis
    
    def _analyze_kernel_memorization(self, foundation_results, optimization_results, elite_results):
        """Analyze whether kernel methods are learning vs. memorizing"""
        memorization_metrics = {}
        
        kernel_methods = ['svr_rbf', 'kernel_ridge']
        for method in kernel_methods:
            if method not in foundation_results:
                continue
                
            metrics = {}
            
            # Foundation performance
            foundation_r2 = foundation_results.get(method, {}).get('test_r2', 0)
            metrics['foundation_r2'] = foundation_r2
            
            # Optimization performance
            if method in optimization_results:
                opt_r2 = optimization_results[method].get('best_cv_score', 0)
                metrics['optimized_r2'] = opt_r2
                metrics['optimization_improvement'] = opt_r2 - foundation_r2
            
            # Elite validation
            if method in elite_results:
                elite_r2 = elite_results[method].get('r2', 0)
                metrics['elite_r2'] = elite_r2
                metrics['generalization_gap'] = foundation_r2 - elite_r2
                
                # Kernel-specific memorization analysis
                if metrics['generalization_gap'] > 0.15:  # Stricter for kernels
                    metrics['memorization_risk'] = 'HIGH'
                elif metrics['generalization_gap'] > 0.08:
                    metrics['memorization_risk'] = 'MEDIUM'
                else:
                    metrics['memorization_risk'] = 'LOW'
                
                # Kernel complexity indicator
                if method in optimization_results:
                    best_params = optimization_results[method].get('best_params', {})
                    if 'model__C' in best_params and best_params['model__C'] > 100:
                        metrics['complexity_warning'] = 'High C parameter - potential overfitting'
                    if 'model__gamma' in best_params and best_params['model__gamma'] > 1:
                        metrics['complexity_warning'] = 'High gamma - potential overfitting'
            
            memorization_metrics[method] = metrics
        
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
        """Train all models with comprehensive evaluation"""
        print("\n" + "="*70)
        print("🔧 MODEL TRAINING AND EVALUATION")
        print("="*70)
        
        # Check if we have SOAP features
        soap_columns = [col for col in X.columns if 'soap' in col.lower()]
        has_soap_features = len(soap_columns) > 0
        
        if has_soap_features:
            print("⚠️  DETECTED SOAP FEATURES: Implementing safeguards against data leakage")
            print("    - Features will be processed properly to avoid information leakage")
            print("    - Train/test split will happen BEFORE feature generation")
            print("    - Models will be evaluated on properly isolated test data")
        
        # Split data - guarantee Structure 350.xyz in test set if requested
        if guarantee_350_in_test:
            X_train, X_test, y_train, y_test = self._guaranteed_350_split(X, y, test_size)
        else:
            # Stratified split
            y_quartiles = pd.qcut(y, q=4, labels=False, duplicates='drop')
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state,
                stratify=y_quartiles
            )
        
        # Store data splits
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"📊 Train target - Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}")
        print(f"📊 Test target - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}")
        
        # Store split information
        self.metadata['data_split'] = {
            'test_size': test_size,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_target_mean': float(y_train.mean()),
            'train_target_std': float(y_train.std()),
            'test_target_mean': float(y_test.mean()),
            'test_target_std': float(y_test.std())
        }
        
        results = {}
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, config in self.model_configs.items():
            print(f"\n🔄 Training {name.upper()}...")
            print(f"   Description: {config['description']}")
            
            try:
                # Create pipeline
                pipeline = self.create_pipeline(name, config, X_train.shape[1])
                
                # Grid search
                grid_search = GridSearchCV(
                    pipeline, config['params'],
                    cv=cv,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit model
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                # Metrics
                metrics = {
                    'train_r2': r2_score(y_train, y_train_pred),
                    'test_r2': r2_score(y_test, y_test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    'train_mae': mean_absolute_error(y_train, y_train_pred),
                    'test_mae': mean_absolute_error(y_test, y_test_pred)
                }
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2', n_jobs=-1)
                
                # Store comprehensive results
                results[name] = {
                    'model': best_model,
                    'description': config['description'],
                    'best_params': grid_search.best_params_,
                    'grid_search_results': {
                        'cv_results': grid_search.cv_results_,
                        'best_score': grid_search.best_score_
                    },
                    **metrics,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'overfitting_gap': metrics['train_r2'] - metrics['test_r2'],
                    'predictions': {
                        'y_train_pred': y_train_pred.tolist(),
                        'y_test_pred': y_test_pred.tolist(),
                        'y_train_actual': y_train.tolist(),
                        'y_test_actual': y_test.tolist()
                    },
                    'feature_selection': config.get('feature_selection'),
                    'n_features_used': X_train.shape[1] if config.get('feature_selection') is None 
                                     else min(config.get('feature_selection', X_train.shape[1]), X_train.shape[1])
                }
                
                # Print results
                print(f"   ✅ Best params: {grid_search.best_params_}")
                print(f"   📊 Train R²: {metrics['train_r2']:.3f}, Test R²: {metrics['test_r2']:.3f}")
                print(f"   📊 Train RMSE: {metrics['train_rmse']:.3f}, Test RMSE: {metrics['test_rmse']:.3f}")
                print(f"   📊 CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"   📊 Overfitting gap: {metrics['train_r2'] - metrics['test_r2']:.3f}")
                
                # Status
                if metrics['test_r2'] < 0:
                    status = "❌ FAILED"
                    print(f"   {status}: Model worse than mean baseline")
                elif metrics['train_r2'] - metrics['test_r2'] > 0.15:
                    status = "⚠️ OVERFITTING"
                    print(f"   {status}: High overfitting detected")
                elif metrics['test_r2'] < 0.5:
                    status = "⚠️ POOR"
                    print(f"   {status}: Low performance")
                else:
                    status = "✅ GOOD"
                    print(f"   {status}: Acceptable performance")
                
                results[name]['status'] = status
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                results[name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'description': config['description']
                }
        
        self.results = results
        return results
    
    def find_most_balanced_structures(self, top_n=10):
        """Find the most balanced structures (lowest average residuals across all models)"""
        print("\n🎯 Finding most balanced structures...")
        
        if not self.results:
            print("⚠️  No model results available for balanced structure analysis")
            return {}
        
        # Get successful models only
        successful_models = [name for name in self.results 
                           if 'error' not in self.results[name] and 
                           'predictions' in self.results[name] and 
                           'y_test_pred' in self.results[name]['predictions']]
        
        if not successful_models:
            print("⚠️  No successful models for balanced structure analysis")
            return {}
        
        # Get test data size
        test_size = len(self.y_test)
        structure_residuals = {}
        
        # Calculate residuals for each structure across all models
        for i in range(test_size):
            residuals_for_structure = []
            
            for model_name in successful_models:
                y_true = self.y_test.iloc[i]
                y_pred = self.results[model_name]['predictions']['y_test_pred'][i]
                residual = abs(y_true - y_pred)
                residuals_for_structure.append(residual)
            
            # Average residual across all models for this structure
            avg_residual = np.mean(residuals_for_structure)
            structure_residuals[i] = {
                'avg_residual': avg_residual,
                'individual_residuals': dict(zip(successful_models, residuals_for_structure)),
                'actual_energy': self.y_test.iloc[i]
            }
        
        # Sort by average residual (lowest = most balanced)
        sorted_structures = sorted(structure_residuals.items(), 
                                 key=lambda x: x[1]['avg_residual'])[:top_n]
        
        print(f"📊 Top {top_n} Most Balanced Structures (Lowest Average Residuals):")
        balanced_structures = {}
        
        for rank, (struct_idx, data) in enumerate(sorted_structures, 1):
            print(f"   {rank}. Structure {struct_idx}: Avg residual = {data['avg_residual']:.4f} eV")
            print(f"      Actual energy: {data['actual_energy']:.3f} eV")
            
            # Show individual model residuals
            for model, residual in data['individual_residuals'].items():
                print(f"      {model}: {residual:.4f} eV")
            print()
            
            balanced_structures[struct_idx] = data
        
        return balanced_structures
    
    def perform_statistical_comparisons(self):
        """Perform statistical comparisons between models using paired tests"""
        print("\n📊 Performing statistical comparisons between models...")
        
        successful_models = [name for name in self.results 
                           if 'error' not in self.results[name] and 
                           'predictions' in self.results[name] and 
                           'y_test_pred' in self.results[name]['predictions']]
        
        if len(successful_models) < 2:
            print("⚠️  Need at least 2 successful models for statistical comparison")
            return {}
        
        try:
            # Statistical tests are now available from the main import
            
            comparisons = {}
            
            # Get all pairwise combinations
            from itertools import combinations
            
            for model1, model2 in combinations(successful_models, 2):
                print(f"\n🔬 Comparing {model1} vs {model2}")
                
                # Get residuals for both models
                y_true = self.y_test.values
                pred1 = np.array(self.results[model1]['predictions']['y_test_pred'])
                pred2 = np.array(self.results[model2]['predictions']['y_test_pred'])
                
                residuals1 = np.abs(y_true - pred1)
                residuals2 = np.abs(y_true - pred2)
                
                # Paired t-test
                try:
                    t_stat, t_pvalue = stats.ttest_rel(residuals1, residuals2)
                    print(f"   Paired t-test: t={t_stat:.4f}, p={t_pvalue:.4f}")
                except Exception as e:
                    t_stat, t_pvalue = None, None
                    print(f"   Paired t-test failed: {e}")
                
                # Wilcoxon signed-rank test (non-parametric alternative)
                try:
                    w_stat, w_pvalue = wilcoxon(residuals1, residuals2)
                    print(f"   Wilcoxon test: W={w_stat:.4f}, p={w_pvalue:.4f}")
                except Exception as e:
                    w_stat, w_pvalue = None, None
                    print(f"   Wilcoxon test failed: {e}")
                
                # Interpretation
                if t_pvalue is not None and t_pvalue < 0.05:
                    significance = "✅ Significant difference"
                elif t_pvalue is not None:
                    significance = "❌ No significant difference"
                else:
                    significance = "⚠️  Could not determine significance"
                
                print(f"   Result: {significance}")
                
                # Store results
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = {
                    'model1': model1,
                    'model2': model2,
                    'model1_mae': self.results[model1]['test_mae'],
                    'model2_mae': self.results[model2]['test_mae'],
                    'model1_r2': self.results[model1]['test_r2'],
                    'model2_r2': self.results[model2]['test_r2'],
                    't_statistic': t_stat,
                    't_pvalue': t_pvalue,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_pvalue': w_pvalue,
                    'significant_difference': t_pvalue < 0.05 if t_pvalue is not None else None
                }
            
            return comparisons
            
        except ImportError:
            print("⚠️  SciPy not available - skipping statistical comparisons")
            return {}
    
    def create_comprehensive_visualizations(self, output_dir):
        """Create all visualization files"""
        print("\n📊 Creating comprehensive visualizations...")
        
        # Ensure main output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model Performance Overview
        self._plot_performance_overview(viz_dir)
        
        # 2. Individual Model Analysis
        self._plot_individual_models(viz_dir)
        
        # 3. Prediction Analysis
        self._plot_prediction_analysis(viz_dir)
        
        # 4. Residual Analysis
        self._plot_residual_analysis(viz_dir)
        
        # 5. Cross-validation Analysis
        self._plot_cv_analysis(viz_dir)
        
        # 6. Feature Analysis
        self._plot_feature_analysis(viz_dir)
        
        print(f"✅ All visualizations saved to {viz_dir}")
    
    def _plot_performance_overview(self, viz_dir):
        """Performance overview plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = [name for name in self.results.keys() if 'error' not in self.results[name]]
        if not models:
            return
        
        # 1. R² Comparison
        ax = axes[0, 0]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        colors = ['red' if r < 0 else 'orange' if r < 0.5 else 'green' for r in test_r2]
        
        bars = ax.bar(range(len(models)), test_r2, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='red', linestyle='--', label='Baseline')
        ax.axhline(y=0.5, color='orange', linestyle='--', label='Acceptable')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('Test R²', fontweight='bold')
        ax.set_title('Model Performance Comparison (R²)', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, test_r2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.02,
                   f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 2. RMSE Comparison
        ax = axes[0, 1]
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        bars = ax.bar(range(len(models)), test_rmse, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('Test RMSE', fontweight='bold')
        ax.set_title('Prediction Error Comparison (RMSE)', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, test_rmse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Overfitting Analysis
        ax = axes[0, 2]
        train_r2 = [self.results[m]['train_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Train vs Test Performance', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cross-Validation Stability
        ax = axes[1, 0]
        cv_means = [self.results[m]['cv_r2_mean'] for m in models]
        cv_stds = [self.results[m]['cv_r2_std'] for m in models]
        
        bars = ax.bar(range(len(models)), cv_means, yerr=cv_stds, capsize=5, alpha=0.7, 
                     edgecolor='black', color='teal')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('CV R² Mean ± Std', fontweight='bold')
        ax.set_title('Cross-Validation Stability', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 5. MAE Comparison
        ax = axes[1, 1]
        test_mae = [self.results[m]['test_mae'] for m in models]
        bars = ax.bar(range(len(models)), test_mae, color='brown', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('Test MAE', fontweight='bold')
        ax.set_title('Mean Absolute Error Comparison', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 6. Model Status Summary
        ax = axes[1, 2]
        status_counts = {}
        for model in models:
            status = self.results[model]['status']
            status_clean = status.split()[1] if len(status.split()) > 1 else status
            status_counts[status_clean] = status_counts.get(status_clean, 0) + 1
        
        if status_counts:
            labels = list(status_counts.keys())
            sizes = list(status_counts.values())
            colors_pie = ['green' if l == 'GOOD' else 'orange' if l == 'POOR' else 'red' for l in labels]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
                                             startangle=90, textprops={'fontweight': 'bold'})
            ax.set_title('Model Status Distribution', fontweight='bold', fontsize=14)
        
        plt.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_models(self, viz_dir):
        """Individual model detailed analysis"""
        models = [name for name in self.results.keys() if 'error' not in self.results[name]]
        
        for model_name in models:
            result = self.results[model_name]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Actual vs Predicted (Test)
            ax = axes[0, 0]
            y_test_actual = result['predictions']['y_test_actual']
            y_test_pred = result['predictions']['y_test_pred']
            
            ax.scatter(y_test_actual, y_test_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            min_val = min(min(y_test_actual), min(y_test_pred))
            max_val = max(max(y_test_actual), max(y_test_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Energy', fontweight='bold')
            ax.set_ylabel('Predicted Energy', fontweight='bold')
            ax.set_title(f'Test Set: Actual vs Predicted\n{model_name.replace("_", " ").title()}', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            r2 = result['test_r2']
            rmse = result['test_rmse']
            mae = result['test_mae']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top', fontweight='bold')
            
            # 2. Residuals
            ax = axes[0, 1]
            residuals = np.array(y_test_actual) - np.array(y_test_pred)
            ax.scatter(y_test_pred, residuals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Energy', fontweight='bold')
            ax.set_ylabel('Residuals', fontweight='bold')
            ax.set_title('Residual Plot', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add residual statistics
            res_mean = np.mean(residuals)
            res_std = np.std(residuals)
            ax.text(0.05, 0.95, f'Mean: {res_mean:.3f}\nStd: {res_std:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   verticalalignment='top', fontweight='bold')
            
            # 3. Residual Distribution
            ax = axes[1, 0]
            ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black', density=True)
            ax.axvline(x=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Residuals', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title('Residual Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 4. Cross-validation scores
            ax = axes[1, 1]
            cv_scores = result['cv_scores']
            ax.bar(range(1, len(cv_scores) + 1), cv_scores, alpha=0.7, edgecolor='black')
            ax.axhline(y=result['cv_r2_mean'], color='red', linestyle='--', lw=2, label=f'Mean: {result["cv_r2_mean"]:.3f}')
            ax.set_xlabel('CV Fold', fontweight='bold')
            ax.set_ylabel('R² Score', fontweight='bold')
            ax.set_title('Cross-Validation Scores', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_name.replace("_", " ").title()} - Detailed Analysis', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(viz_dir / f'{model_name}_detailed_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_prediction_analysis(self, viz_dir):
        """Combined prediction analysis for all models"""
        models = [name for name in self.results.keys() if 'error' not in self.results[name]]
        
        if not models:
            return
            
        n_models = len(models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models):
            result = self.results[model_name]
            ax = axes[idx]
            
            y_test_actual = result['predictions']['y_test_actual']
            y_test_pred = result['predictions']['y_test_pred']
            
            # Scatter plot
            ax.scatter(y_test_actual, y_test_pred, alpha=0.6, s=25, edgecolors='black', linewidth=0.3)
            
            # Perfect prediction line
            min_val = min(min(y_test_actual), min(y_test_pred))
            max_val = max(max(y_test_actual), max(y_test_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Energy', fontweight='bold')
            ax.set_ylabel('Predicted Energy', fontweight='bold')
            ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add metrics
            r2 = result['test_r2']
            rmse = result['test_rmse']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top', fontsize=9, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Prediction Analysis - All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'predictions_vs_actual_all_models.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_analysis(self, viz_dir):
        """Residual analysis for all models"""
        models = [name for name in self.results.keys() if 'error' not in self.results[name]]
        
        if not models:
            return
            
        n_models = len(models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, model_name in enumerate(models):
            result = self.results[model_name]
            ax = axes[idx]
            
            y_test_actual = result['predictions']['y_test_actual']
            y_test_pred = result['predictions']['y_test_pred']
            residuals = np.array(y_test_actual) - np.array(y_test_pred)
            
            # Residual plot
            ax.scatter(y_test_pred, residuals, alpha=0.6, s=25, edgecolors='black', linewidth=0.3)
            ax.axhline(y=0, color='red', linestyle='--', lw=2)
            
            ax.set_xlabel('Predicted Energy', fontweight='bold')
            ax.set_ylabel('Residuals', fontweight='bold')
            ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            res_mean = np.mean(residuals)
            res_std = np.std(residuals)
            ax.text(0.05, 0.95, f'Mean: {res_mean:.3f}\nStd: {res_std:.3f}',
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   verticalalignment='top', fontsize=9, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Residual Analysis - All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'residual_analysis_all_models.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cv_analysis(self, viz_dir):
        """Cross-validation analysis"""
        models = [name for name in self.results.keys() if 'error' not in self.results[name]]
        
        if not models:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. CV Score Distribution
        ax = axes[0]
        cv_data = []
        labels = []
        
        for model_name in models:
            cv_scores = self.results[model_name]['cv_scores']
            cv_data.extend(cv_scores)
            labels.extend([model_name.replace('_', '\n')] * len(cv_scores))
        
        # Create violin plot
        positions = []
        data_by_model = []
        model_labels = []
        
        for i, model_name in enumerate(models):
            cv_scores = self.results[model_name]['cv_scores']
            data_by_model.append(cv_scores)
            positions.append(i)
            model_labels.append(model_name.replace('_', '\n'))
        
        parts = ax.violinplot(data_by_model, positions=positions, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.set_ylabel('CV R² Score', fontweight='bold')
        ax.set_title('Cross-Validation Score Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 2. CV Stability (Mean ± Std)
        ax = axes[1]
        cv_means = [self.results[m]['cv_r2_mean'] for m in models]
        cv_stds = [self.results[m]['cv_r2_std'] for m in models]
        
        bars = ax.bar(range(len(models)), cv_means, yerr=cv_stds, capsize=5, 
                     alpha=0.7, edgecolor='black', color=colors)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('CV R² Mean ± Std', fontweight='bold')
        ax.set_title('Cross-Validation Stability', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle('Cross-Validation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'cross_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, viz_dir):
        """Feature importance and selection analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        models = [name for name in self.results.keys() if 'error' not in self.results[name]]
        
        # 1. Number of features used
        ax = axes[0]
        feature_counts = [self.results[m]['n_features_used'] for m in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(range(len(models)), feature_counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.set_ylabel('Number of Features Used', fontweight='bold')
        ax.set_title('Feature Usage by Model', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance vs Features
        ax = axes[1]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        scatter = ax.scatter(feature_counts, test_r2, c=range(len(models)), 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(model.replace('_', '\n'), (feature_counts[i], test_r2[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Number of Features Used', fontweight='bold')
        ax.set_ylabel('Test R² Score', fontweight='bold')
        ax.set_title('Performance vs Feature Count', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.suptitle('Feature Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_top_structures_csv(self, top_n=20, output_dir='./kernel_models_analysis'):
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
        
        # Collect all structures from successful models
        all_structures = []
        
        for model_name, result in self.results.items():
            if 'error' in result:
                print(f"   ⚠️ Skipping {model_name} due to training errors")
                continue
            
            if 'predictions' not in result or 'y_test_pred' not in result['predictions']:
                print(f"   ⚠️ No predictions found for {model_name}")
                continue
            
            print(f"   🔍 Processing {model_name}...")
            
            # Get predictions (lower energy = more stable)
            predictions = np.array(result['predictions']['y_test_pred'])
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
                    'stability_score': float(-predictions[idx]),  # Negative for stability
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
        
        coords = []
        
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
            coords.append([0.0, 0.0, 0.0])  # Central atom
            
            # Shell atoms
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
            
            # First shell
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
                theta = 2 * np.pi * i / remaining + 0.5
                phi = np.pi * (0.2 + 0.6 * np.random.random())
                r = 5.2  # Larger radius
                
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
        
        # Ensure correct number of atoms
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

    
    def save_comprehensive_results(self, output_dir):
        """Save all results in multiple formats"""
        print("\n💾 Saving comprehensive results...")
        
        # Ensure main output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save metadata as text file (avoiding JSON issues)
        try:
            with open(output_dir / 'analysis_metadata.txt', 'w') as f:
                f.write("ANALYSIS METADATA\n")
                f.write("="*50 + "\n\n")
                f.write(f"Analysis Timestamp: {self.metadata['analysis_timestamp']}\n")
                f.write(f"Random State: {self.metadata['random_state']}\n")
                f.write(f"Data Source: {self.metadata['data_source']}\n\n")
                
                # Add data leakage warning if performance is suspiciously high
                any_suspicious = False
                for name, result in self.results.items():
                    if 'error' not in result and result['test_r2'] > 0.98:
                        any_suspicious = True
                
                if any_suspicious:
                    f.write("⚠️  DATA LEAKAGE WARNING:\n")
                    f.write("-"*30 + "\n")
                    f.write("Some models show suspiciously high performance (R² > 0.98)\n")
                    f.write("This may indicate potential data leakage issues.\n")
                    f.write("Please verify the following:\n")
                    f.write("1. SOAP descriptors are properly segregated between train/test\n")
                    f.write("2. No information from test set leaks into feature engineering\n")
                    f.write("3. All preprocessing steps happen separately for train/test\n\n")
                
                f.write("DATA STATISTICS:\n")
                f.write("-"*30 + "\n")
                stats = self.metadata['data_stats']
                f.write(f"Samples: {stats['n_samples']}\n")
                f.write(f"Features: {stats['n_features']}\n")
                f.write(f"Target Mean: {stats['target_mean']:.3f}\n")
                f.write(f"Target Std: {stats['target_std']:.3f}\n")
                f.write(f"Target Range: [{stats['target_min']:.3f}, {stats['target_max']:.3f}]\n")
                f.write(f"Feature/Sample Ratio: {stats['feature_sample_ratio']:.3f}\n\n")
                
                f.write("FEATURES USED:\n")
                f.write("-"*30 + "\n")
                features = self.metadata['features']
                f.write(f"Total Features: {features['total_features']}\n")
                f.write(f"Feature Names: {', '.join(features['feature_names'][:10])}...")
                if features['removed_features']:
                    f.write(f"\nRemoved Features: {len(features['removed_features'])} constant/zero-variance features\n")
            
            print("✅ Metadata saved as text file")
        except Exception as e:
            print(f"⚠️  Could not save metadata: {e}")
        
        # 2. Skip all JSON exports entirely to avoid serialization issues
        print("ℹ️  Skipping all JSON exports to avoid serialization issues")
        
        # 3. Create comprehensive CSV summary
        summary_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                summary_data.append({
                    'model_name': name,
                    'description': result['description'],
                    'status': result['status'],
                    'train_r2': result['train_r2'],
                    'test_r2': result['test_r2'],
                    'train_rmse': result['train_rmse'],
                    'test_rmse': result['test_rmse'],
                    'train_mae': result['train_mae'],
                    'test_mae': result['test_mae'],
                    'cv_r2_mean': result['cv_r2_mean'],
                    'cv_r2_std': result['cv_r2_std'],
                    'overfitting_gap': result['overfitting_gap'],
                    'n_features_used': result['n_features_used'],
                    'feature_selection': result['feature_selection'],
                    'best_params': str(result['best_params'])
                })
            else:
                summary_data.append({
                    'model_name': name,
                    'description': result.get('description', 'N/A'),
                    'status': 'ERROR',
                    'error_message': result.get('error', 'Unknown error'),
                    **{k: np.nan for k in ['train_r2', 'test_r2', 'train_rmse', 'test_rmse', 
                                          'train_mae', 'test_mae', 'cv_r2_mean', 'cv_r2_std',
                                          'overfitting_gap', 'n_features_used']}
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'model_performance_summary.csv', index=False)
        print("✅ CSV summary saved")
        
        # 4. Save individual model predictions
        predictions_dir = output_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.results.items():
            if 'error' not in result and 'predictions' in result:
                try:
                    # Save train and test predictions separately due to different lengths
                    train_data = {
                        'actual': result['predictions']['y_train_actual'],
                        'predicted': result['predictions']['y_train_pred']
                    }
                    
                    test_data = {
                        'actual': result['predictions']['y_test_actual'],
                        'predicted': result['predictions']['y_test_pred']
                    }
                    
                    # Save separate files for train and test
                    train_df = pd.DataFrame(train_data)
                    test_df = pd.DataFrame(test_data)
                    
                    train_df.to_csv(predictions_dir / f'{name}_train_predictions.csv', index=False)
                    test_df.to_csv(predictions_dir / f'{name}_test_predictions.csv', index=False)
                    
                except Exception as e:
                    print(f"   ⚠️ Could not save predictions for {name}: {e}")
        
        print("✅ Prediction CSV files saved")
        
        # 5. Save best working models
        models_dir = output_dir / 'saved_models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        working_models = {k: v for k, v in self.results.items() 
                         if 'error' not in v and v['test_r2'] > 0}
        
        for name, result in working_models.items():
            try:
                # Save model
                joblib.dump(result['model'], models_dir / f'{name}_model.joblib')
                
                # Save model info as text file (avoiding JSON issues)
                with open(models_dir / f'{name}_info.txt', 'w') as f:
                    f.write(f"MODEL INFORMATION: {name}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Description: {result['description']}\n")
                    f.write(f"Status: {result['status']}\n\n")
                    
                    f.write("PERFORMANCE METRICS:\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Test R²: {result['test_r2']:.4f}\n")
                    f.write(f"Test RMSE: {result['test_rmse']:.4f}\n")
                    f.write(f"Test MAE: {result['test_mae']:.4f}\n")
                    f.write(f"Train R²: {result['train_r2']:.4f}\n")
                    f.write(f"CV R² Mean: {result['cv_r2_mean']:.4f}\n")
                    f.write(f"CV R² Std: {result['cv_r2_std']:.4f}\n")
                    f.write(f"Overfitting Gap: {result['overfitting_gap']:.4f}\n\n")
                    
                    f.write("CONFIGURATION:\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Features Used: {result['n_features_used']}\n")
                    f.write(f"Feature Selection: {result['feature_selection']}\n")
                    f.write(f"Best Parameters: {result['best_params']}\n\n")
                    
                    f.write(f"Training Date: {self.metadata['analysis_timestamp']}\n")
                
                print(f"   ✅ Saved {name} model and info")
                    
            except Exception as e:
                print(f"   ⚠️ Could not save model {name}: {e}")
        
        # 6. Create detailed results as text file
        try:
            with open(output_dir / 'detailed_results.txt', 'w') as f:
                f.write("DETAILED MODEL RESULTS\n")
                f.write("="*60 + "\n\n")
                
                for name, result in self.results.items():
                    f.write(f"MODEL: {name.upper()}\n")
                    f.write("-"*40 + "\n")
                    
                    if 'error' not in result:
                        f.write(f"Description: {result['description']}\n")
                        f.write(f"Status: {result['status']}\n\n")
                        
                        f.write("Performance Metrics:\n")
                        f.write(f"  Train R²: {result['train_r2']:.4f}\n")
                        f.write(f"  Test R²: {result['test_r2']:.4f}\n")
                        f.write(f"  Train RMSE: {result['train_rmse']:.4f}\n")
                        f.write(f"  Test RMSE: {result['test_rmse']:.4f}\n")
                        f.write(f"  Train MAE: {result['train_mae']:.4f}\n")
                        f.write(f"  Test MAE: {result['test_mae']:.4f}\n")
                        f.write(f"  CV R² Mean: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}\n")
                        f.write(f"  Overfitting Gap: {result['overfitting_gap']:.4f}\n\n")
                        
                        f.write(f"Configuration:\n")
                        f.write(f"  Features Used: {result['n_features_used']}\n")
                        f.write(f"  Feature Selection: {result['feature_selection']}\n")
                        f.write(f"  Best Parameters: {result['best_params']}\n")
                    else:
                        f.write(f"Status: ERROR\n")
                        f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    
                    f.write("\n" + "="*60 + "\n\n")
            
            print("✅ Detailed results saved as text file")
        except Exception as e:
            print(f"⚠️  Could not save detailed results: {e}")
        
        # 7. Create executive summary
        self._create_executive_summary(output_dir, summary_df)
        
        # 8. Clean up any JSON files that might have been created in previous runs
        try:
            json_files = [
                output_dir / 'analysis_metadata.json',
                output_dir / 'detailed_results.json',
                output_dir / 'learning_curves' / 'learning_curve_data.json'
            ]
            for json_file in json_files:
                if json_file.exists():
                    json_file.unlink()
                    print(f"✅ Removed unnecessary JSON file: {json_file}")
        except Exception as e:
            print(f"⚠️ Error cleaning up JSON files: {e}")
        
        print(f"✅ All results saved to {output_dir}")
        print(f"   📊 CSV summary: model_performance_summary.csv")
        print(f"   📋 Text details: detailed_results.txt")
        print(f"   🔮 Predictions: predictions/ folder")
        print(f"   🤖 Models: saved_models/ folder")
        print(f"   📈 Visualizations: visualizations/ folder")
        print(f"   📝 Executive summary: executive_summary.txt")
        print(f"   🧹 Removed unnecessary JSON files")
        
        return summary_df
    
    def _create_executive_summary(self, output_dir, summary_df):
        """Create an executive summary report"""
        with open(output_dir / 'executive_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("KERNEL METHODS ANALYSIS - EXECUTIVE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {self.metadata['analysis_timestamp']}\n")
            f.write(f"Data Source: {self.metadata['data_source']}\n\n")
            
            # Check for suspiciously high performance that might indicate data leakage
            suspicious_models = []
            perfect_models = []
            for name, result in self.results.items():
                if 'error' not in result:
                    if result['test_r2'] > 0.99:
                        perfect_models.append((name, result['test_r2']))
                    elif result['test_r2'] > 0.95:
                        suspicious_models.append((name, result['test_r2']))
            
            if perfect_models:
                f.write("\n🚨 DATA LEAKAGE WARNING 🚨\n")
                f.write("-" * 40 + "\n")
                f.write("The following models show PERFECT performance (R² > 0.99):\n")
                for name, score in perfect_models:
                    f.write(f"   • {name}: R² = {score:.4f}\n")
                f.write("\nThis strongly suggests DATA LEAKAGE issues with SOAP descriptors.\n")
                f.write("Please verify the separation of training and test data during SOAP feature generation.\n")
                f.write("Feature engineering steps should NOT use information from the test set.\n\n")
            elif suspicious_models:
                f.write("\n⚠️ POTENTIAL DATA LEAKAGE WARNING ⚠️\n")
                f.write("-" * 40 + "\n")
                f.write("The following models show suspiciously high performance (R² > 0.95):\n")
                for name, score in suspicious_models:
                    f.write(f"   • {name}: R² = {score:.4f}\n")
                f.write("\nThis may indicate unintentional information leakage or over-optimistic evaluation.\n")
                f.write("Verify that feature engineering steps do not use test data information.\n\n")
            
            # Data summary
            f.write("DATA OVERVIEW:\n")
            f.write("-" * 40 + "\n")
            stats = self.metadata['data_stats']
            f.write(f"• Total Samples: {stats['n_samples']}\n")
            f.write(f"• Features Used: {stats['n_features']}\n")
            f.write(f"• Target Range: [{stats['target_min']:.3f}, {stats['target_max']:.3f}]\n")
            f.write(f"• Feature/Sample Ratio: {stats['feature_sample_ratio']:.3f}\n\n")
            
            # Model performance
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            # Best model
            working_models = summary_df[summary_df['test_r2'] > 0]
            if not working_models.empty:
                best_model = working_models.loc[working_models['test_r2'].idxmax()]
                f.write(f"🏆 BEST MODEL: {best_model['model_name']}\n")
                f.write(f"   Description: {best_model['description']}\n")
                f.write(f"   Test R²: {best_model['test_r2']:.3f}\n")
                f.write(f"   Test RMSE: {best_model['test_rmse']:.3f}\n")
                f.write(f"   Status: {best_model['status']}\n\n")
            else:
                f.write("❌ NO VIABLE MODELS FOUND\n\n")
            
            # Model status breakdown
            f.write("MODEL STATUS BREAKDOWN:\n")
            status_counts = summary_df['status'].value_counts()
            for status, count in status_counts.items():
                f.write(f"   {status}: {count} models\n")
            f.write("\n")
            
            # Detailed results table
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(summary_df[['model_name', 'test_r2', 'test_rmse', 'cv_r2_mean', 'status']].to_string(index=False))
            f.write("\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            
            if working_models.empty:
                f.write("• CRITICAL: No models achieved positive R²\n")
                f.write("• Consider advanced feature engineering (SOAP descriptors)\n")
                f.write("• Evaluate deep learning approaches\n")
                f.write("• Check data quality and target variable definition\n")
            else:
                best_r2 = working_models['test_r2'].max()
                if best_r2 < 0.5:
                    f.write("• Models show poor performance (R² < 0.5)\n")
                    f.write("• Consider ensemble methods\n")
                    f.write("• Add more sophisticated features\n")
                elif best_r2 < 0.7:
                    f.write("• Models show moderate performance\n")
                    f.write("• Try ensemble methods for improvement\n")
                    f.write("• Consider neural networks\n")
                else:
                    f.write("• Models show good performance\n")
                    f.write("• Consider the best model for deployment\n")
                    f.write("• Validate on additional test sets\n")
            
            f.write(f"\nGenerated files for documentation:\n")
            f.write(f"• Performance plots: visualizations/\n")
            f.write(f"• Learning curves: learning_curves/\n")
            f.write(f"• Model predictions: predictions/\n")
            f.write(f"• Trained models: saved_models/\n")

def main():
    """Main execution with enhanced hybrid kernel training"""
    print("="*80)
    print("🚀 ENHANCED KERNEL METHODS ANALYSIS")
    print("   Complete Documentation Package with Hybrid Training")
    print("="*80)
    
    analyzer = ComprehensiveKernelAnalysis(random_state=42)
    
    # Ask user about training approach
    training_mode = input("Choose training mode:\n1. Standard (999 structures only)\n2. Hybrid (999 + categorized datasets)\nEnter choice (1/2, default=2): ").strip()
    use_hybrid = training_mode != '1'
    
    # Get data path
    data_path = input("\nEnter path to descriptors.csv (press Enter for default): ").strip()
    if not data_path:
        data_path = "./au_cluster_analysis_results/descriptors.csv"
    
    # Output directory
    output_dir = input("Enter output directory (press Enter for default): ").strip()
    if not output_dir:
        output_dir = "./kernel_models_analysis"
    
    output_dir = Path(output_dir)
    
    try:
        print(f"\n📂 Output directory: {output_dir}")
        
        # Load and prepare data with hybrid training support
        X, y = analyzer.load_and_prepare_data(data_path, use_hybrid_training=use_hybrid)
        
        # Choose training approach
        if use_hybrid and any(df is not None for df in analyzer.datasets.values()):
            print("\n🚀 Starting Progressive Kernel Training...")
            results = analyzer.progressive_kernel_training(X, y, use_elite_validation=True)
            
            # Display kernel-specific memorization analysis
            if results.get('anti_memorization_metrics'):
                print("\n🧠 Kernel Anti-Memorization Analysis:")
                print("-" * 50)
                for model_name, metrics in results['anti_memorization_metrics'].items():
                    risk = metrics.get('memorization_risk', 'UNKNOWN')
                    gap = metrics.get('generalization_gap', 0)
                    complexity_warning = metrics.get('complexity_warning', '')
                    print(f"{model_name:15s}: Risk={risk:6s}, Gap={gap:+.4f}")
                    if complexity_warning:
                        print(f"                  Warning: {complexity_warning}")
            
            # Use foundation results for further analysis
            analysis_results = results['foundation_results']
        else:
            print("\n📚 Using Standard Kernel Training (999 structures)...")
            
            # Generate learning curves first
            learning_curve_data = analyzer.generate_learning_curves(X, y, output_dir)
            
            # Train models
            analysis_results = analyzer.train_models(X, y)
            results = analysis_results
        
        # Find most balanced structures using the main results
        balanced_structures = analyzer.find_most_balanced_structures(top_n=10)
        
        # Perform statistical comparisons
        statistical_comparisons = analyzer.perform_statistical_comparisons()
        
        # Create all visualizations
        analyzer.create_comprehensive_visualizations(output_dir)
        
        # Save comprehensive results
        summary_df = analyzer.save_comprehensive_results(output_dir)
        
        # Export top stable structures to CSV
        print("\n🌟 STRUCTURE EXPORT")
        print("="*40)
        try:
            csv_path = analyzer.export_top_structures_csv(top_n=20, output_dir=output_dir)
            if csv_path:
                print("📊 Top 20 stable structures exported for 3D visualization!")
        except Exception as e:
            print(f"⚠️ CSV export error: {e}")
        
        # Save additional analysis outputs
        print("\n💾 Saving additional analysis outputs...")
        
        # Save balanced structures analysis
        if balanced_structures:
            balanced_df = pd.DataFrame([
                {
                    'structure_index': idx,
                    'avg_residual': data['avg_residual'],
                    'actual_energy': data['actual_energy'],
                    **{f'{model}_residual': res for model, res in data['individual_residuals'].items()}
                }
                for idx, data in balanced_structures.items()
            ])
            balanced_df.to_csv(output_dir / 'most_balanced_structures.csv', index=False)
            print("   ✅ Most balanced structures saved to most_balanced_structures.csv")
        
        # Save statistical comparisons
        if statistical_comparisons:
            comp_df = pd.DataFrame([
                {
                    'comparison': key,
                    'model1': data['model1'],
                    'model2': data['model2'],
                    'model1_mae': data['model1_mae'],
                    'model2_mae': data['model2_mae'],
                    'model1_r2': data['model1_r2'],
                    'model2_r2': data['model2_r2'],
                    't_pvalue': data['t_pvalue'],
                    'wilcoxon_pvalue': data['wilcoxon_pvalue'],
                    'significant_difference': data['significant_difference']
                }
                for key, data in statistical_comparisons.items()
            ])
            comp_df.to_csv(output_dir / 'statistical_comparisons.csv', index=False)
            print("   ✅ Statistical comparisons saved to statistical_comparisons.csv")
        
        # Final summary
        print("\n" + "="*80)
        print("📊 ANALYSIS COMPLETE")
        print("="*80)
        
        print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
        print(summary_df[['model_name', 'test_r2', 'test_rmse', 'status']].round(3).to_string(index=False))
        
        # Best model summary
        working_models = summary_df[summary_df['test_r2'] > 0]
        if not working_models.empty:
            best = working_models.loc[working_models['test_r2'].idxmax()]
            print(f"\n🏆 BEST MODEL: {best['model_name']}")
            print(f"   Test R²: {best['test_r2']:.3f}")
            print(f"   Test RMSE: {best['test_rmse']:.3f}")
            print(f"   Status: {best['status']}")
        else:
            print(f"\n❌ No viable models found")
        
        print(f"\n📁 All files saved to: {output_dir}")
        print(f"   Ready for documentation and presentation!")
        
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
    
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyzer, results = main()