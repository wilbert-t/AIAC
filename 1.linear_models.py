#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
from matplotlib.patches import Patch  # Added for legend in feature importance
warnings.filterwarnings('ignore')

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
    Linear & Regularized Models for Au Cluster Analysis
    
    Why Linear Models for Au Clusters:
    1. Interpretability: Direct coefficient interpretation shows feature importance
    2. Baseline Performance: Establishes minimum expected performance
    3. Feature Selection: Lasso identifies most important SOAP/structural features
    4. Computational Efficiency: Fast training and prediction
    5. Regularization: Handles multicollinearity in SOAP descriptors
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.soap_features = None
        self.feature_names = None  # Added to store actual feature names
        
        # Initialize models with justifications
        self.model_configs = {
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
    
    def load_data(self, data_path, target_column='energy'):
        """Load data from task1.py output"""
        if isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        else:
            self.df = data_path  # DataFrame passed directly
        
        # Clean data
        self.df = self.df.dropna(subset=[target_column])
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Target range: {self.df[target_column].min():.2f} to {self.df[target_column].max():.2f}")
        
        return self.df
    
    def create_soap_features(self, structures_data=None):
        """
        Create SOAP descriptors for enhanced accuracy
        
        Why SOAP for Au Clusters:
        - Rotationally invariant descriptors
        - Captures local atomic environments
        - Smooth, differentiable features
        - Proven performance for molecular ML
        """
        if not SOAP_AVAILABLE or structures_data is None:
            print("Using basic descriptors only")
            return None
        
        print("Creating SOAP descriptors for enhanced accuracy...")
        
        # SOAP parameters optimized for Au clusters
        soap = SOAP(
            species=['Au'],
            r_cut=5.0,      # Au-Au interaction cutoff
            n_max=8,        # Radial basis functions
            l_max=6,        # Angular basis functions  
            sigma=0.5,      # Gaussian smearing
            periodic=False, # Clusters (not crystals)
            sparse=False,   # Dense output for ML
            average='inner' # Average over atoms
        )
        
        soap_features = []
        filenames = []
        
        for structure in structures_data:
            try:
                atoms = structure['atoms'] if 'atoms' in structure else None
                if atoms is None:
                    # Create atoms object from coordinates
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
            
            print(f"Added {soap_array.shape[1]} SOAP features")
            self.soap_features = [col for col in self.df.columns if col.startswith('soap_')]
            
        return self.soap_features
    
    def prepare_features(self, target_column='energy', include_soap=True):
        """Prepare feature matrix and target vector"""
        # Select features
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
        
        # Add SOAP features if available
        if include_soap and self.soap_features:
            feature_cols.extend(self.soap_features)
            print(f"Using {len(self.soap_features)} SOAP features")
        
        # Remove any remaining NaN values
        feature_cols = [f for f in feature_cols if f in self.df.columns]
        data_clean = self.df[feature_cols + [target_column]].dropna()
        
        X = data_clean[feature_cols]
        y = data_clean[target_column]
        
        self.feature_names = feature_cols  # Store the feature names
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Using features: {len(feature_cols)} total")
        
        return X, y, feature_cols
    
    def train_models(self, X, y, test_size=0.2):
        """Train all linear models"""
        print("\n" + "="*60)
        print("TRAINING LINEAR & REGULARIZED MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Store splits
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        results = {}
        
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
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                Pipeline([('scaler', StandardScaler()), ('model', config['model'])]),
                X, y, cv=5, scoring='r2'
            )
            
            results[name] = {
                'model': model,
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
    
    def analyze_feature_importance(self, feature_names):
        """Analyze feature importance from linear models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        importance_data = []
        
        for name, result in self.results.items():
            if hasattr(result['model'], 'coef_'):
                coeffs = result['model'].coef_
                
                # For regularized models, non-zero coefficients are selected features
                if hasattr(result['model'], 'alpha_'):
                    alpha = result['model'].alpha_
                    print(f"\n{name.upper()} (α = {alpha:.4f}):")
                    
                    # Count non-zero coefficients
                    non_zero = np.sum(np.abs(coeffs) > 1e-6)
                    print(f"  Selected features: {non_zero}/{len(coeffs)}")
                    
                    # Top features by absolute coefficient
                    abs_coeffs = np.abs(coeffs)
                    top_indices = np.argsort(abs_coeffs)[::-1][:10]
                    
                    for i, idx in enumerate(top_indices):
                        if abs_coeffs[idx] > 1e-6:
                            print(f"  {i+1:2d}. {feature_names[idx]:<25} | {coeffs[idx]:8.4f}")
                            
                            importance_data.append({
                                'model': name,
                                'feature': feature_names[idx],
                                'coefficient': coeffs[idx],
                                'abs_coefficient': abs_coeffs[idx],
                                'rank': i + 1
                            })
        
        # Create importance DataFrame
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            return importance_df
        return None
    
    def create_visualizations(self, output_dir='./linear_models_results'):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(output_dir)
        
        # 2. Prediction vs Actual plots
        self._plot_predictions(output_dir)
        
        # 3. Residual analysis
        self._plot_residuals(output_dir)
        
        # 4. Feature importance (for regularized models)
        self._plot_feature_importance(output_dir)
        
        print(f"📊 Visualizations saved to {output_dir}")
    
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
        axes[0,0].set_title('Model R² Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # RMSE
        train_rmse = [self.results[m]['train_rmse'] for m in models]
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        
        axes[0,1].bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
        axes[0,1].bar(x + width/2, test_rmse, width, label='Test', alpha=0.8)
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].set_title('Model RMSE Performance Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(models)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Cross-validation scores
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        
        axes[1,0].bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        axes[1,0].set_ylabel('CV R² Score')
        axes[1,0].set_title('Cross-Validation Performance Comparison')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(models)
        axes[1,0].grid(True, alpha=0.3)
        
        # MAE comparison
        test_mae = [self.results[m]['test_mae'] for m in models]
        axes[1,1].bar(x, test_mae, alpha=0.8)
        axes[1,1].set_ylabel('Mean Absolute Error')
        axes[1,1].set_title('Model MAE Performance Comparison (Test Set)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(models)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot predicted vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
                
            # Test set predictions
            y_true = self.y_test
            y_pred = result['y_test_pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # R² annotation
            r2 = result['test_r2']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel('Actual Energy')
            axes[i].set_ylabel('Predicted Energy')
            axes[i].set_title(f'{name.title()} Predictions vs Actual (Test Set)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residuals(self, output_dir):
        """Plot residual analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
                
            y_pred = result['y_test_pred']
            residuals = self.y_test - y_pred
            
            axes[i].scatter(y_pred, residuals, alpha=0.6, s=50)
            axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.8, label='Zero Residual')
            
            axes[i].set_xlabel('Predicted Energy')
            axes[i].set_ylabel('Residuals (Actual - Predicted)')
            axes[i].set_title(f'{name.title()} Residual Analysis (Test Set)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, output_dir):
        """Plot feature importance for regularized models"""
        if self.feature_names is None:
            print("Feature names not available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        regularized_models = ['ridge', 'lasso', 'elastic_net']
        
        for i, name in enumerate(regularized_models):
            if name not in self.results or i >= 4:
                continue
                
            result = self.results[name]
            if hasattr(result['model'], 'coef_'):
                coeffs = result['model'].coef_
                
                # Get top 15 features by absolute coefficient
                abs_coeffs = np.abs(coeffs)
                top_indices = np.argsort(abs_coeffs)[::-1][:15]
                
                top_coeffs = coeffs[top_indices]
                top_feature_names = [self.feature_names[j] for j in top_indices]
                
                colors = ['red' if c < 0 else 'blue' for c in top_coeffs]
                
                axes[i].barh(range(len(top_coeffs)), top_coeffs, color=colors, alpha=0.7)
                axes[i].set_yticks(range(len(top_coeffs)))
                axes[i].set_yticklabels(top_feature_names, fontsize=8)
                axes[i].set_xlabel('Coefficient Value')
                axes[i].set_title(f'{name.title()} Top Feature Coefficients')
                axes[i].grid(True, alpha=0.3)
                axes[i].invert_yaxis()
                
                # Add legend for colors
                legend_elements = [Patch(facecolor='blue', edgecolor='blue', label='Positive Coefficient'),
                                   Patch(facecolor='red', edgecolor='red', label='Negative Coefficient')]
                axes[i].legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, output_dir='./linear_models_results'):
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
        summary_df.to_csv(output_dir / 'model_summary.csv', index=False)
        
        print(f"💾 Models and results saved to {output_dir}")
        
        return summary_df

def main():
    """Main execution function"""
    print("🔬 Linear & Regularized Models for Au Cluster Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = LinearModelsAnalyzer(random_state=42)
    
    # Load data (assuming task1.py has been run)
    try:
        data_path = input("Enter path to descriptors.csv (default: ./au_cluster_analysis_results/descriptors.csv): ").strip()
        if not data_path:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        analyzer.load_data(data_path)
        
        # Prepare features
        X, y, feature_names = analyzer.prepare_features(target_column='energy')
        
        # Train models
        results = analyzer.train_models(X, y)
        
        # Analyze feature importance
        importance_df = analyzer.analyze_feature_importance(feature_names)
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_models()
        
        print("\n🎉 Linear models analysis complete!")
        print("\nBest performing model:")
        best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
        print(f"  {best_model['model'].upper()}: R² = {best_model['test_r2']:.3f}")
        
        return analyzer, results
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    analyzer, results = main()