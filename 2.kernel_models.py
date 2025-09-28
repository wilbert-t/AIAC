#!/usr/bin/env python3
"""
ENHANCED: Kernel Methods for Au Cluster Energy Prediction
Complete documentation package with all files and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
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
    Complete Kernel Methods Analysis with Full Documentation Package
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
            'ridge_regression': {
                'model': Ridge(random_state=random_state),
                'params': {
                    'model__alpha': [0.1, 1, 10, 100, 1000]  # Stronger regularization for SOAP
                },
                'feature_selection': None,
                'description': 'Linear regression with L2 regularization'
            },
            
            'elastic_net': {
                'model': ElasticNet(random_state=random_state, max_iter=5000),
                'params': {
                    'model__alpha': [0.1, 1.0, 10.0, 50.0],  # Higher regularization
                    'model__l1_ratio': [0.1, 0.5, 0.9]       # Simplified grid
                },
                'feature_selection': None,
                'description': 'Linear regression with L1+L2 regularization'
            },
            
            'svr_rbf_conservative': {
                'model': SVR(kernel='rbf', cache_size=1000),
                'params': {
                    'model__C': [1, 10, 100],                # Conservative C values
                    'model__gamma': [0.001, 0.01, 0.1],      # Conservative gamma
                    'model__epsilon': [0.1, 0.5, 1.0]       # Larger epsilon for stability
                },
                'feature_selection': 25,  # More features for SOAP
                'description': 'Support Vector Regression with RBF kernel'
            },
            
            'svr_linear': {
                'model': SVR(kernel='linear', cache_size=1000),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__epsilon': [0.1, 0.5, 1.0]
                },
                'feature_selection': None,
                'description': 'Support Vector Regression with linear kernel'
            },
            
            'kernel_ridge_rbf': {
                'model': KernelRidge(kernel='rbf'),
                'params': {
                    'model__alpha': [1, 10, 100, 1000],       # Stronger regularization
                    'model__gamma': [0.001, 0.01, 0.1]       # Conservative gamma
                },
                'feature_selection': 20,  # Feature selection for stability
                'description': 'Kernel Ridge Regression with RBF kernel'
            },
            
            'kernel_ridge_linear': {
                'model': KernelRidge(kernel='linear'),
                'params': {
                    'model__alpha': [10, 100, 1000, 10000]   # Much stronger regularization
                },
                'feature_selection': None,
                'description': 'Kernel Ridge Regression with linear kernel'
            },
            
            'knn_stable': {
                'model': KNeighborsRegressor(n_jobs=-1),
                'params': {
                    'model__n_neighbors': [5, 10, 15, 20],    # Smaller neighborhoods
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan']
                },
                'feature_selection': 15,  # Feature selection for KNN
                'description': 'K-Nearest Neighbors Regression'
            }
        }
    
    def create_soap_features(self, structures_data=None, n_components=100):
        """
        Create SOAP (Smooth Overlap of Atomic Positions) descriptors
        
        Args:
            structures_data: List of ASE Atoms objects or coordinates
            n_components: Number of SOAP components to generate
            
        Returns:
            DataFrame with SOAP features
        """
        if not SOAP_AVAILABLE:
            print("⚠️  SOAP features not available - DScribe not installed")
            return None
            
        if structures_data is None:
            print("⚠️  No structure data provided for SOAP features")
            return None
        
        print("🧪 Generating SOAP descriptors...")
        
        # SOAP parameters optimized for Au clusters
        soap = SOAP(
            species=["Au"],
            rcut=6.0,  # Cutoff radius in Angstroms
            nmax=8,    # Maximum radial basis functions
            lmax=6,    # Maximum degree of spherical harmonics
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
            return None
        
        # Convert to DataFrame
        soap_array = np.array(soap_features)
        
        # Apply PCA to reduce dimensionality if needed
        if soap_array.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            soap_reduced = pca.fit_transform(soap_array)
            
            feature_names = [f'soap_pca_{i}' for i in range(n_components)]
            print(f"📊 SOAP features reduced from {soap_array.shape[1]} to {n_components} via PCA")
            print(f"📊 PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            soap_reduced = soap_array
            feature_names = [f'soap_{i}' for i in range(soap_reduced.shape[1])]
        
        soap_df = pd.DataFrame(soap_reduced, columns=feature_names, index=valid_indices)
        
        print(f"✅ Generated {len(soap_df)} SOAP feature vectors with {soap_df.shape[1]} components")
        
        return soap_df
    
    def load_and_prepare_data(self, data_path, target_column='energy'):
        """Load and prepare data with comprehensive logging"""
        print("\n" + "="*70)
        print("📁 DATA LOADING AND PREPARATION")
        print("="*70)
        
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
        
        # Get numeric features only
        exclude_cols = [target_column, 'filename', 'Unnamed: 0']
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
        soap_features = None
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
                    soap_df = self.create_soap_features(structures, n_components=50)
                    if soap_df is not None:
                        # Align SOAP features with main dataframe
                        if len(soap_df) == len(X_basic):
                            X_combined = pd.concat([X_basic.reset_index(drop=True), 
                                                  soap_df.reset_index(drop=True)], axis=1)
                            soap_features = list(soap_df.columns)
                            print(f"✅ Combined {len(valid_features)} basic + {len(soap_features)} SOAP features")
                        else:
                            print("⚠️  SOAP feature count mismatch - using basic features only")
                            X_combined = X_basic
                    else:
                        X_combined = X_basic
                else:
                    print("⚠️  No valid structures found - using basic features only")
                    X_combined = X_basic
            else:
                print("⚠️  No coordinate data found - using basic features only")
                X_combined = X_basic
                
        except Exception as e:
            print(f"⚠️  SOAP feature generation failed: {e}")
            X_combined = X_basic
        
        X = X_combined
        
        # Store feature information
        all_features = valid_features + (soap_features if soap_features else [])
        self.metadata['features'] = {
            'total_features': len(all_features),
            'basic_features': len(valid_features),
            'soap_features': len(soap_features) if soap_features else 0,
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
        
        # Save learning curve data (with proper serialization)
        try:
            with open(learning_curves_dir / 'learning_curve_data.json', 'w') as f:
                json.dump(learning_curve_data, f, indent=2)
            print("✅ Learning curve data saved")
        except Exception as e:
            print(f"⚠️  Could not save learning curve JSON: {e}")
            print("   Learning curves plots still available...")
        
        return learning_curve_data
    
    def train_models(self, X, y, test_size=0.2):
        """Train all models with comprehensive evaluation"""
        print("\n" + "="*70)
        print("🔧 MODEL TRAINING AND EVALUATION")
        print("="*70)
        
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
        
        # 2. Skip JSON export entirely to avoid serialization issues
        print("ℹ️  Skipping JSON export to avoid serialization issues")
        
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
                    # Convert numpy arrays to lists for safe handling
                    pred_data = {
                        'train_actual': result['predictions']['y_train_actual'],
                        'train_predicted': result['predictions']['y_train_pred'],
                        'test_actual': result['predictions']['y_test_actual'],
                        'test_predicted': result['predictions']['y_test_pred']
                    }
                    
                    # Convert to DataFrame (pandas handles numpy arrays automatically)
                    pred_df = pd.DataFrame(pred_data)
                    pred_df.to_csv(predictions_dir / f'{name}_predictions.csv', index=False)
                    
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
        
        print(f"✅ All results saved to {output_dir}")
        print(f"   📊 CSV summary: model_performance_summary.csv")
        print(f"   📋 Text details: detailed_results.txt")
        print(f"   🔮 Predictions: predictions/ folder")
        print(f"   🤖 Models: saved_models/ folder")
        print(f"   📈 Visualizations: visualizations/ folder")
        print(f"   📝 Executive summary: executive_summary.txt")
        
        return summary_df
    
    def _create_executive_summary(self, output_dir, summary_df):
        """Create an executive summary report"""
        with open(output_dir / 'executive_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("KERNEL METHODS ANALYSIS - EXECUTIVE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {self.metadata['analysis_timestamp']}\n")
            f.write(f"Data Source: {self.metadata['data_source']}\n\n")
            
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
    """Main execution with comprehensive analysis"""
    print("="*80)
    print("🚀 ENHANCED KERNEL METHODS ANALYSIS")
    print("   Complete Documentation Package")
    print("="*80)
    
    analyzer = ComprehensiveKernelAnalysis(random_state=42)
    
    # Get data path
    data_path = input("\nEnter path to descriptors.csv (press Enter for default): ").strip()
    if not data_path:
        data_path = "./au_cluster_analysis_results/descriptors.csv"
    
    # Output directory
    output_dir = input("Enter output directory (press Enter for default): ").strip()
    if not output_dir:
        output_dir = "./comprehensive_kernel_analysis"
    
    output_dir = Path(output_dir)
    
    try:
        print(f"\n📂 Output directory: {output_dir}")
        
        # Load and prepare data
        X, y = analyzer.load_and_prepare_data(data_path)
        
        # Generate learning curves first
        learning_curve_data = analyzer.generate_learning_curves(X, y, output_dir)
        
        # Train models
        results = analyzer.train_models(X, y)
        
        # Create all visualizations
        analyzer.create_comprehensive_visualizations(output_dir)
        
        # Save comprehensive results
        summary_df = analyzer.save_comprehensive_results(output_dir)
        
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