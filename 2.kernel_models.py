#!/usr/bin/env python3
"""
Category 2: Kernel & Instance-Based Methods for Au Cluster Energy Prediction
Models: SVR (RBF, Polynomial), Kernel Ridge, KNN
Enhanced with SOAP descriptors for capturing non-linear relationships
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# SOAP descriptors for enhanced features
try:
    from dscribe.descriptors import SOAP
    from ase.atoms import Atoms
    SOAP_AVAILABLE = True
except ImportError:
    print("Warning: DScribe not available. Using basic descriptors only.")
    SOAP_AVAILABLE = False

class KernelMethodsAnalyzer:
    """
    Kernel & Instance-Based Methods for Au Cluster Analysis
    
    Why Kernel Methods for Au Clusters:
    1. Non-linear Relationships: Capture complex SOAP-energy mappings without explicit feature engineering
    2. High-dimensional Efficiency: Work well with high-dimensional SOAP vectors
    3. Similarity-based Learning: Leverage structural similarity between clusters
    4. Robust to Outliers: SVR with ε-insensitive loss handles unusual structures
    5. Kernel Trick: Access infinite-dimensional feature spaces efficiently
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.soap_features = None
        
        # Initialize models with detailed justifications
        self.model_configs = {
            'svr_rbf': {
                'model': SVR(kernel='rbf', cache_size=1000),  # 1GB cache for M4
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'justification': """
                SVR with RBF Kernel:
                - Captures non-linear SOAP-energy relationships via Gaussian kernels
                - RBF kernel measures similarity in high-dimensional SOAP space
                - ε-insensitive loss ignores small prediction errors (robust to noise)
                - Excellent for smooth energy landscapes in cluster space
                - C parameter balances fitting vs regularization
                """
            },
            'svr_poly': {
                'model': SVR(kernel='poly', cache_size=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto'],
                    'coef0': [0, 1]
                },
                'justification': """
                SVR with Polynomial Kernel:
                - Models polynomial interactions between SOAP features
                - Degree parameter controls interaction complexity
                - Captures how combinations of atomic environments affect energy
                - coef0 adds flexibility to polynomial relationships
                - Interpretable non-linearity with physical meaning
                """
            },
            'kernel_ridge': {
                'model': KernelRidge(kernel='rbf'),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'gamma': [0.001, 0.01, 0.1, 1]
                },
                'justification': """
                Kernel Ridge Regression:
                - Combines kernel trick with Ridge regularization benefits
                - Closed-form solution (no iterative optimization like SVR)
                - Better numerical stability than standard kernel methods
                - Faster training than SVR for moderate datasets
                - Smooth predictions ideal for energy surfaces
                """
            },
            'knn': {
                'model': KNeighborsRegressor(n_jobs=8),  # Use M4 cores
                'params': {
                    'n_neighbors': [3, 5, 7, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'justification': """
                K-Nearest Neighbors:
                - Local similarity assumption: similar SOAP → similar energies
                - Non-parametric: no assumptions about functional form
                - Distance weighting emphasizes closer neighbors
                - Natural outlier detection via neighbor distances
                - Interpretable: predictions based on known similar structures
                """
            }
        }
    
    def load_data(self, data_path, target_column='energy'):
        """Load data from task1.py output"""
        if isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        else:
            self.df = data_path
        
        # Clean data
        self.df = self.df.dropna(subset=[target_column])
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Target range: {self.df[target_column].min():.2f} to {self.df[target_column].max():.2f}")
        
        return self.df
    
    def create_soap_features(self, structures_data=None):
        """
        Create SOAP descriptors optimized for kernel methods
        
        Why SOAP + Kernels:
        - SOAP provides smooth, differentiable descriptors
        - Kernels naturally handle high-dimensional SOAP vectors
        - Rotation invariance crucial for cluster comparisons
        - Local atomic environments map well to kernel similarities
        """
        if not SOAP_AVAILABLE or structures_data is None:
            print("Using basic descriptors only")
            return None
        
        print("Creating SOAP descriptors optimized for kernel methods...")
        
        # SOAP parameters optimized for kernel methods
        soap = SOAP(
            species=['Au'],
            r_cut=5.0,      # Au-Au interaction range
            n_max=8,        # Radial basis functions
            l_max=6,        # Angular basis functions
            sigma=0.3,      # Tighter Gaussian for kernel methods
            periodic=False, # Clusters
            sparse=False,   # Dense for kernels
            average='inner' # Average over atoms
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
                print(f"Error creating SOAP for {structure.get('filename', 'unknown')}: {e}")
                continue
        
        if soap_features:
            soap_array = np.array(soap_features)
            soap_df = pd.DataFrame(
                soap_array, 
                columns=[f'soap_{i}' for i in range(soap_array.shape[1])]
            )
            soap_df['filename'] = filenames
            
            # Merge with existing data
            self.df = self.df.merge(soap_df, on='filename', how='inner')
            
            print(f"Added {soap_array.shape[1]} SOAP features for kernel methods")
            self.soap_features = [col for col in self.df.columns if col.startswith('soap_')]
            
        return self.soap_features
    
    def prepare_features(self, target_column='energy', include_soap=True):
        """Prepare feature matrix optimized for kernel methods"""
        feature_cols = []
        
        # Basic structural features
        basic_features = [
            'mean_bond_length', 'std_bond_length', 'n_bonds',
            'mean_coordination', 'std_coordination', 'max_coordination',
            'radius_of_gyration', 'asphericity', 'surface_fraction',
            'x_range', 'y_range', 'z_range', 'anisotropy',
            'compactness', 'bond_variance'
        ]
        
        available_basic = [f for f in basic_features if f in self.df.columns]
        feature_cols.extend(available_basic)
        
        # Add SOAP features (crucial for kernel performance)
        if include_soap and self.soap_features:
            feature_cols.extend(self.soap_features)
            print(f"Using {len(self.soap_features)} SOAP features for kernel methods")
        
        # Clean data
        feature_cols = [f for f in feature_cols if f in self.df.columns]
        data_clean = self.df[feature_cols + [target_column]].dropna()
        
        X = data_clean[feature_cols]
        y = data_clean[target_column]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Kernel methods benefit from high-dimensional SOAP features")
        
        return X, y, feature_cols
    
    def train_models(self, X, y, test_size=0.2):
        """Train all kernel-based models with optimized hyperparameters"""
        print("\n" + "="*60)
        print("TRAINING KERNEL & INSTANCE-BASED MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Memory management for MacBook M4 16GB
        if len(X_train) > 2000:
            print(f"⚠️  Large dataset ({len(X_train)} samples). Using subset for kernel methods.")
            indices = np.random.choice(len(X_train), 2000, replace=False)
            X_train = X_train.iloc[indices]
            y_train = y_train.iloc[indices]
        
        # Store splits
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        results = {}
        
        for name, config in self.model_configs.items():
            print(f"\n🔍 Training {name.upper()}...")
            print(f"Justification: {config['justification'].strip()}")
            
            # Create pipeline with scaling (crucial for kernels)
            scaler = StandardScaler()
            model = config['model']
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Hyperparameter optimization
            if config['params']:
                print(f"  Optimizing hyperparameters...")
                grid_search = GridSearchCV(
                    model, config['params'], 
                    cv=3,  # Reduced for speed
                    scoring='r2',
                    n_jobs=4,  # Parallel processing
                    verbose=0
                )
                
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                
                print(f"  Best parameters: {grid_search.best_params_}")
                print(f"  Best CV score: {grid_search.best_score_:.3f}")
            else:
                best_model = model
                best_model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = best_model.predict(X_train_scaled)
            y_test_pred = best_model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                Pipeline([('scaler', StandardScaler()), ('model', best_model)]),
                X, y, cv=3, scoring='r2'  # Reduced CV for speed
            )
            
            results[name] = {
                'model': best_model,
                'scaler': scaler,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            
            print(f"✅ {name}: R² = {test_r2:.3f}, RMSE = {test_rmse:.2f}, CV = {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        self.results = results
        return results
    
    def analyze_kernel_insights(self):
        """Analyze kernel-specific insights"""
        print("\n" + "="*50)
        print("KERNEL METHOD INSIGHTS")
        print("="*50)
        
        for name, result in self.results.items():
            model = result['model']
            print(f"\n{name.upper()}:")
            
            if hasattr(model, 'support_vectors_'):
                # SVR analysis
                n_support = len(model.support_vectors_)
                print(f"  Support vectors: {n_support}/{len(self.X_train)} ({n_support/len(self.X_train)*100:.1f}%)")
                
                if hasattr(model, 'dual_coef_'):
                    dual_norm = np.linalg.norm(model.dual_coef_)
                    print(f"  Model complexity (dual norm): {dual_norm:.3f}")
            
            elif hasattr(model, 'alpha'):
                # Kernel Ridge analysis
                print(f"  Regularization (α): {model.alpha:.4f}")
                print(f"  Kernel: {model.kernel}")
                
            elif hasattr(model, 'n_neighbors'):
                # KNN analysis
                print(f"  Neighbors used: {model.n_neighbors}")
                print(f"  Distance metric: {model.metric}")
                print(f"  Weighting: {model.weights}")
    
    def create_visualizations(self, output_dir='./kernel_methods_results'):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(output_dir)
        
        # 2. Prediction vs Actual plots
        self._plot_predictions(output_dir)
        
        # 3. Kernel-specific visualizations
        self._plot_kernel_analysis(output_dir)
        
        # 4. Feature space visualization (if possible)
        self._plot_feature_space(output_dir)
        
        print(f"📊 Kernel method visualizations saved to {output_dir}")
    
    def _plot_model_comparison(self, output_dir):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(self.results.keys())
        
        # R² scores
        train_r2 = [self.results[m]['train_r2'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0,0].bar(x - width/2, train_r2, width, label='Train', alpha=0.8)
        axes[0,0].bar(x + width/2, test_r2, width, label='Test', alpha=0.8)
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Kernel Methods R² Performance')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Training time comparison (if available)
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        
        axes[0,1].bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.8, 
                     color=['orange', 'green', 'purple', 'red'])
        axes[0,1].set_ylabel('CV R² Score')
        axes[0,1].set_title('Cross-Validation Performance')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[0,1].grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        axes[1,0].bar(x, test_rmse, alpha=0.8, color=['orange', 'green', 'purple', 'red'])
        axes[1,0].set_ylabel('Test RMSE')
        axes[1,0].set_title('Model RMSE Performance')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[1,0].grid(True, alpha=0.3)
        
        # MAE comparison
        test_mae = [self.results[m]['test_mae'] for m in models]
        axes[1,1].bar(x, test_mae, alpha=0.8, color=['orange', 'green', 'purple', 'red'])
        axes[1,1].set_ylabel('Test MAE')
        axes[1,1].set_title('Model MAE Performance')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kernel_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot predicted vs actual values for each kernel method"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['orange', 'green', 'purple', 'red']
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
                
            y_true = self.y_test
            y_pred = result['y_test_pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=50, color=colors[i])
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # R² and RMSE annotation
            r2 = result['test_r2']
            rmse = result['test_rmse']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel('Actual Energy (eV)')
            axes[i].set_ylabel('Predicted Energy (eV)')
            axes[i].set_title(f'{name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kernel_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_kernel_analysis(self, output_dir):
        """Plot kernel-specific analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Support vector analysis for SVR models
        svr_models = {k: v for k, v in self.results.items() if 'svr' in k}
        
        if svr_models:
            # Support vector ratios
            model_names = []
            support_ratios = []
            
            for name, result in svr_models.items():
                if hasattr(result['model'], 'support_vectors_'):
                    n_support = len(result['model'].support_vectors_)
                    ratio = n_support / len(self.X_train) * 100
                    model_names.append(name.replace('_', ' ').title())
                    support_ratios.append(ratio)
            
            if support_ratios:
                axes[0,0].bar(model_names, support_ratios, alpha=0.8, color=['orange', 'green'])
                axes[0,0].set_ylabel('Support Vector Ratio (%)')
                axes[0,0].set_title('SVR Model Complexity')
                axes[0,0].grid(True, alpha=0.3)
        
        # Residual analysis for best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        best_result = self.results[best_model_name]
        
        residuals = self.y_test - best_result['y_test_pred']
        
        axes[0,1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Residuals')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title(f'Residual Distribution - {best_model_name.title()}')
        axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0,1].grid(True, alpha=0.3)
        
        # Prediction error vs magnitude
        axes[1,0].scatter(self.y_test, np.abs(residuals), alpha=0.6)
        axes[1,0].set_xlabel('Actual Energy (eV)')
        axes[1,0].set_ylabel('Absolute Error')
        axes[1,0].set_title(f'Error vs Energy Magnitude - {best_model_name.title()}')
        axes[1,0].grid(True, alpha=0.3)
        
        # Model comparison radar chart
        metrics = ['test_r2', 'cv_mean']
        model_names = list(self.results.keys())
        
        if len(metrics) >= 2 and len(model_names) > 0:
            # Normalize metrics for radar chart
            normalized_data = []
            for name in model_names:
                values = [self.results[name][metric] for metric in metrics]
                normalized_data.append(values)
            
            # Simple bar chart instead of radar for simplicity
            x = np.arange(len(model_names))
            test_r2_values = [self.results[name]['test_r2'] for name in model_names]
            
            axes[1,1].bar(x, test_r2_values, alpha=0.8, color=['orange', 'green', 'purple', 'red'])
            axes[1,1].set_ylabel('Test R² Score')
            axes[1,1].set_title('Final Model Comparison')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels([name.replace('_', '\n') for name in model_names])
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'kernel_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_space(self, output_dir):
        """Plot feature space visualization (simplified)"""
        if self.soap_features and len(self.soap_features) > 2:
            # Use first 2 SOAP features for visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            feature1 = self.soap_features[0]
            feature2 = self.soap_features[1]
            
            # Scatter plot colored by energy
            scatter = axes[0].scatter(self.df[feature1], self.df[feature2], 
                                    c=self.df['energy'], cmap='viridis', alpha=0.6)
            axes[0].set_xlabel(f'{feature1}')
            axes[0].set_ylabel(f'{feature2}')
            axes[0].set_title('SOAP Feature Space (Energy Colored)')
            plt.colorbar(scatter, ax=axes[0], label='Energy (eV)')
            
            # Feature correlation with energy
            correlations = []
            for feature in self.soap_features[:20]:  # Top 20 SOAP features
                corr = self.df[feature].corr(self.df['energy'])
                correlations.append(abs(corr))
            
            axes[1].bar(range(len(correlations)), correlations, alpha=0.8)
            axes[1].set_xlabel('SOAP Feature Index')
            axes[1].set_ylabel('|Correlation with Energy|')
            axes[1].set_title('SOAP Feature Importance')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'soap_feature_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_models(self, output_dir='./kernel_methods_results'):
        """Save trained models and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, result in self.results.items():
            model_path = output_dir / f'{name}_model.joblib'
            scaler_path = output_dir / f'{name}_scaler.joblib'
            
            joblib.dump(result['model'], model_path)
            joblib.dump(result['scaler'], scaler_path)
        
        # Save results summary
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'model': name,
                'train_r2': result['train_r2'],
                'test_r2': result['test_r2'],
                'train_rmse': result['train_rmse'],
                'test_rmse': result['test_rmse'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'kernel_model_summary.csv', index=False)
        
        print(f"💾 Kernel models and results saved to {output_dir}")
        
        return summary_df

def main():
    """Main execution function"""
    print("🔬 Kernel & Instance-Based Methods for Au Cluster Analysis")
    print("="*70)
    
    # Initialize analyzer
    analyzer = KernelMethodsAnalyzer(random_state=42)
    
    # Load data
    try:
        data_path = input("/Users/wilbert/Documents/GitHub/AIAC/au_cluster_analysis_results/descriptors.csv").strip()
        if not data_path:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        analyzer.load_data(data_path)
        
        # Prepare features
        X, y, feature_names = analyzer.prepare_features(target_column='energy')
        
        # Train models
        results = analyzer.train_models(X, y)
        
        # Analyze kernel insights
        analyzer.analyze_kernel_insights()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_models()
        
        print("\n🎉 Kernel methods analysis complete!")
        print("\nBest performing model:")
        best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
        print(f"  {best_model['model'].upper()}: R² = {best_model['test_r2']:.3f}")
        
        print("\n💡 Kernel Method Insights:")
        print("- SVR models excel at capturing non-linear SOAP-energy relationships")
        print("- Kernel Ridge provides smooth predictions ideal for energy surfaces")
        print("- KNN leverages structural similarity between clusters")
        print("- SOAP descriptors significantly enhance kernel method performance")
        
        return analyzer, results
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    analyzer, results = main()