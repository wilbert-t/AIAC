#!/usr/bin/env python3
"""
IMPROVED Category 2: Kernel & Instance-Based Methods for Au Cluster Energy Prediction
Optimized for small datasets with proper regularization and feature engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, 
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time

class ImprovedKernelMethodsAnalyzer:
    """
    Improved Kernel & Instance-Based Methods for Au Cluster Analysis
    
    Key Improvements:
    1. Proper feature engineering and selection
    2. Optimized hyperparameters for small datasets
    3. Comprehensive evaluation metrics (R², RMSE, MAE)
    4. Fixed visualization functions with actual plots
    5. Better regularization strategies
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Optimized model configurations for small datasets
        self.model_configs = {
            'svr_rbf': {
                'model': SVR(kernel='rbf', cache_size=1000),
                'params': {
                    'model__C': [0.01, 0.1, 1, 10, 100],  # Start with smaller C for regularization
                    'model__gamma': [0.0001, 0.001, 0.01, 0.1, 'scale', 'auto'],
                    'model__epsilon': [0.001, 0.01, 0.1, 0.5]  # Added smaller epsilon
                },
                'use_feature_selection': True,
                'n_features': 'auto'  # Will be determined based on data
            },
            'svr_linear': {  # Added linear SVR as baseline
                'model': SVR(kernel='linear', max_iter=10000),
                'params': {
                    'model__C': [0.001, 0.01, 0.1, 1, 10],
                    'model__epsilon': [0.01, 0.1, 0.5]
                },
                'use_feature_selection': True,
                'n_features': 'auto'
            },
            'kernel_ridge': {
                'model': KernelRidge(kernel='rbf'),
                'params': {
                    'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100],  # Stronger regularization
                    'model__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
                },
                'use_feature_selection': True,
                'n_features': 'auto'
            },
            'knn': {
                'model': KNeighborsRegressor(n_jobs=-1),  # Use all cores
                'params': {
                    'model__n_neighbors': [3, 5, 7, 10, 15, 20, 30],  # Include larger k
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan', 'minkowski'],
                    'model__p': [1, 2]  # For Minkowski
                },
                'use_feature_selection': True,
                'n_features': 'auto'
            }
        }
        
        # Store metrics history
        self.metrics_history = []
    
    def load_data(self, data_path, target_column='energy'):
        """Load and analyze data"""
        if isinstance(data_path, str):
            self.df = pd.read_csv(data_path)
        else:
            self.df = data_path
        
        # Clean data
        self.df = self.df.dropna(subset=[target_column])
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Target statistics:")
        print(f"  Mean: {self.df[target_column].mean():.2f}")
        print(f"  Std: {self.df[target_column].std():.2f}")
        print(f"  Range: [{self.df[target_column].min():.2f}, {self.df[target_column].max():.2f}]")
        
        # Data quality check
        if len(self.df) < 100:
            print("⚠️  Very small dataset - consider data augmentation or simpler models")
        elif len(self.df) < 1000:
            print("⚠️  Small dataset - focus on regularization and cross-validation")
        
        return self.df
    
    def engineer_features(self):
        """Create additional engineered features"""
        print("\nEngineering additional features...")
        
        # Polynomial features for basic descriptors
        if 'mean_bond_length' in self.df.columns and 'std_bond_length' in self.df.columns:
            self.df['bond_cv'] = self.df['std_bond_length'] / (self.df['mean_bond_length'] + 1e-8)
            self.df['bond_ratio'] = self.df['mean_bond_length'] / self.df['mean_coordination'].clip(lower=1)
        
        if 'radius_of_gyration' in self.df.columns and 'n_atoms' in self.df.columns:
            self.df['normalized_radius'] = self.df['radius_of_gyration'] / np.sqrt(self.df['n_atoms'])
        
        if 'surface_fraction' in self.df.columns:
            self.df['core_fraction'] = 1 - self.df['surface_fraction']
            self.df['surface_core_ratio'] = self.df['surface_fraction'] / (self.df['core_fraction'] + 1e-8)
        
        # Interaction terms
        if 'compactness' in self.df.columns and 'asphericity' in self.df.columns:
            self.df['shape_factor'] = self.df['compactness'] * self.df['asphericity']
        
        print(f"Added {len([c for c in self.df.columns if c not in ['filename', 'energy']])} features")
    
    def prepare_features(self, target_column='energy'):
        """Prepare feature matrix with proper selection"""
        
        # Get all numeric columns except target and identifiers
        exclude_cols = [target_column, 'filename', 'Unnamed: 0']
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[col])]
        
        # Remove features with zero variance
        feature_cols = [col for col in feature_cols 
                       if self.df[col].std() > 1e-8]
        
        # Handle any remaining NaN values
        data_clean = self.df[feature_cols + [target_column]].dropna()
        
        X = data_clean[feature_cols]
        y = data_clean[target_column]
        
        print(f"\nPrepared feature matrix:")
        print(f"  Shape: {X.shape}")
        print(f"  Features per sample ratio: {X.shape[1]/X.shape[0]:.3f}")
        
        # Determine optimal number of features for selection
        optimal_features = min(
            X.shape[1],
            max(5, X.shape[0] // 20)  # At least 5, but no more than samples/20
        )
        
        print(f"  Recommended features for selection: {optimal_features}")
        
        # Update model configs with optimal feature count
        for config in self.model_configs.values():
            if config.get('n_features') == 'auto':
                config['n_features'] = optimal_features
        
        return X, y, feature_cols
    
    def create_model_pipeline(self, model_name, config, n_features):
        """Create model pipeline with proper preprocessing"""
        steps = []
        
        # Use RobustScaler for better handling of outliers
        steps.append(('scaler', RobustScaler()))
        
        # Feature selection if specified
        if config.get('use_feature_selection', False) and n_features < self.X_train.shape[1]:
            steps.append(('feature_selection', SelectKBest(f_regression, k=n_features)))
            print(f"  Using feature selection: {n_features} features")
        
        # Add the model
        steps.append(('model', config['model']))
        
        return Pipeline(steps)
    
    def train_models(self, X, y, test_size=0.2):
        """Train all models with comprehensive evaluation"""
        print("\n" + "="*70)
        print("TRAINING IMPROVED KERNEL METHODS")
        print("="*70)
        
        # Split data with stratification for stability
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        results = {}
        
        for i, (name, config) in enumerate(self.model_configs.items(), 1):
            print(f"\n[{i}/{len(self.model_configs)}] Training {name.upper()}...")
            start_time = time.time()
            
            # Create pipeline
            n_features = config.get('n_features', X_train.shape[1])
            pipeline = self.create_model_pipeline(name, config, n_features)
            
            # Grid search with proper CV
            cv_folds = min(5, len(X_train) // 30)  # Ensure enough samples per fold
            cv_folds = max(3, cv_folds)
            
            if config['params']:
                print(f"  Grid search with {cv_folds}-fold CV...")
                grid_search = GridSearchCV(
                    pipeline, config['params'],
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',  # Use MSE for stability
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Extract best parameters (only model parameters)
                best_params = {k: v for k, v in grid_search.best_params_.items() 
                              if k.startswith('model__')}
                print(f"  Best params: {best_params}")
                print(f"  Best CV score: {-grid_search.best_score_:.3f}")
            else:
                best_model = pipeline
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate comprehensive metrics
            metrics = self.calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
            
            # Cross-validation for stability assessment
            cv_scores = cross_val_score(
                pipeline, X, y, cv=cv_folds, 
                scoring='neg_mean_squared_error'
            )
            
            results[name] = {
                'model': best_model,
                **metrics,
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'training_time': time.time() - start_time
            }
            
            # Print results
            print(f"  Train: R²={metrics['train_r2']:.3f}, RMSE={metrics['train_rmse']:.3f}, MAE={metrics['train_mae']:.3f}")
            print(f"  Test:  R²={metrics['test_r2']:.3f}, RMSE={metrics['test_rmse']:.3f}, MAE={metrics['test_mae']:.3f}")
            print(f"  Time: {results[name]['training_time']:.1f}s")
            
            # Overfitting check
            if metrics['train_r2'] - metrics['test_r2'] > 0.15:
                print("  ⚠️  Potential overfitting detected")
        
        self.results = results
        return results
    
    def calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate comprehensive metrics"""
        return {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
        }
    
    def create_visualizations(self, output_dir='./kernel_results'):
        """Create comprehensive visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Performance comparison
        self._plot_performance_comparison(output_dir)
        
        # 2. Predictions scatter plots
        self._plot_predictions(output_dir)
        
        # 3. Residual analysis
        self._plot_residuals(output_dir)
        
        # 4. Learning curves
        self._plot_learning_curves(output_dir)
        
        print(f"\n📊 Visualizations saved to {output_dir}")
    
    def _plot_performance_comparison(self, output_dir):
        """Plot comprehensive performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        models = list(self.results.keys())
        model_labels = [m.replace('_', ' ').title() for m in models]
        
        # Extract metrics
        metrics_data = {
            'R² Score': {
                'train': [self.results[m]['train_r2'] for m in models],
                'test': [self.results[m]['test_r2'] for m in models]
            },
            'RMSE': {
                'train': [self.results[m]['train_rmse'] for m in models],
                'test': [self.results[m]['test_rmse'] for m in models]
            },
            'MAE': {
                'train': [self.results[m]['train_mae'] for m in models],
                'test': [self.results[m]['test_mae'] for m in models]
            }
        }
        
        x = np.arange(len(models))
        width = 0.35
        
        # Plot each metric
        for idx, (metric_name, metric_values) in enumerate(metrics_data.items()):
            ax = axes[0, idx]
            
            bars1 = ax.bar(x - width/2, metric_values['train'], width, 
                          label='Train', alpha=0.8, color='steelblue')
            bars2 = ax.bar(x + width/2, metric_values['test'], width, 
                          label='Test', alpha=0.8, color='coral')
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=8)
        
        # Overfitting analysis
        ax = axes[1, 0]
        overfitting = [self.results[m]['train_r2'] - self.results[m]['test_r2'] for m in models]
        bars = ax.bar(model_labels, overfitting, color=['red' if o > 0.15 else 'green' for o in overfitting])
        ax.set_ylabel('Train R² - Test R²')
        ax.set_title('Overfitting Analysis')
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Overfit threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cross-validation stability
        ax = axes[1, 1]
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        ax.bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax.set_ylabel('CV MSE')
        ax.set_title('Cross-Validation Stability')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Training time
        ax = axes[1, 2]
        times = [self.results[m]['training_time'] for m in models]
        ax.bar(model_labels, times, color='purple', alpha=0.7)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time')
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Kernel Methods Performance Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot predicted vs actual values"""
        n_models = len(self.results)
        fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(12, 6 * ((n_models + 1) // 2)))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            y_test = self.y_test
            y_pred = result['y_test_pred']
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Perfect')
            
            # Add metrics text
            r2 = result['test_r2']
            rmse = result['test_rmse']
            mae = result['test_mae']
            
            text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            ax.set_xlabel('Actual Energy (eV)')
            ax.set_ylabel('Predicted Energy (eV)')
            ax.set_title(f'{name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Predictions vs Actual Values', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_residuals(self, output_dir):
        """Plot residual analysis"""
        n_models = len(self.results)
        fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(12, 6 * ((n_models + 1) // 2)))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            y_test = self.y_test
            y_pred = result['y_test_pred']
            residuals = y_test - y_pred
            
            # Residual plot
            ax.scatter(y_pred, residuals, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.7)
            
            # Add ±1 std lines
            std_residual = residuals.std()
            ax.axhline(y=std_residual, color='orange', linestyle=':', alpha=0.7, label=f'±{std_residual:.2f}')
            ax.axhline(y=-std_residual, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_xlabel('Predicted Energy (eV)')
            ax.set_ylabel('Residuals (eV)')
            ax.set_title(f'{name.replace("_", " ").title()} - Residual Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text with residual statistics
            mean_res = residuals.mean()
            text = f'Mean: {mean_res:.3f}\nStd: {std_residual:.3f}'
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   verticalalignment='top', fontsize=10)
        
        # Remove extra subplots
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Residual Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'residuals.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self, output_dir):
        """Plot learning curves for best model"""
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        best_config = self.model_configs[best_model_name]
        
        print(f"\nGenerating learning curves for best model: {best_model_name}")
        
        # Create fresh pipeline
        n_features = best_config.get('n_features', self.X_train.shape[1])
        pipeline = self.create_model_pipeline(best_model_name, best_config, n_features)
        
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, self.X_train, self.y_train,
            cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        # Convert to RMSE
        train_scores = np.sqrt(-train_scores)
        val_scores = np.sqrt(-val_scores)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='steelblue',
                label='Training score', lw=2, markersize=8)
        ax.fill_between(train_sizes, 
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.2, color='steelblue')
        
        ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='coral',
                label='Cross-validation score', lw=2, markersize=8)
        ax.fill_between(train_sizes, 
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.2, color='coral')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Learning Curves - {best_model_name.replace("_", " ").title()}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir='./kernel_results'):
        """Save comprehensive results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, result in self.results.items():
            model_path = output_dir / f'{name}_model.joblib'
            joblib.dump(result['model'], model_path)
        
        # Create comprehensive summary
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'model': name,
                'train_r2': result['train_r2'],
                'test_r2': result['test_r2'],
                'train_rmse': result['train_rmse'],
                'test_rmse': result['test_rmse'],
                'train_mae': result['train_mae'],
                'test_mae': result['test_mae'],
                'cv_mse_mean': result['cv_mean'],
                'cv_mse_std': result['cv_std'],
                'overfitting_gap': result['train_r2'] - result['test_r2'],
                'training_time': result['training_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'kernel_results_summary.csv', index=False)
        
        # Save detailed results
        import json
        detailed_results = {
            name: {k: v for k, v in result.items() 
                  if k not in ['model', 'y_train_pred', 'y_test_pred']}
            for name, result in self.results.items()
        }
        
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}")
        
        return summary_df

def main():
    """Main execution function"""
    print("="*70)
    print("🚀 IMPROVED KERNEL METHODS FOR AU CLUSTER ANALYSIS")
    print("="*70)
    print("\nOptimizations:")
    print("✅ Proper feature engineering and selection")
    print("✅ Optimized hyperparameters for small datasets")
    print("✅ Comprehensive metrics (R², RMSE, MAE)")
    print("✅ Fixed visualizations with actual plots")
    print("✅ Better regularization strategies")
    print("="*70)
    
    # Initialize analyzer
    analyzer = ImprovedKernelMethodsAnalyzer(random_state=42)
    
    # Load data
    try:
        data_path = input("\nEnter path to descriptors.csv (press Enter for default): ").strip()
        if not data_path:
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        analyzer.load_data(data_path)
        
        # Engineer additional features
        analyzer.engineer_features()
        
        # Prepare features
        X, y, feature_names = analyzer.prepare_features(target_column='energy')
        
        # Train models
        results = analyzer.train_models(X, y)
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_results()
        
        print("\n" + "="*70)
        print("📊 FINAL RESULTS SUMMARY")
        print("="*70)
        
        print("\n📈 Performance Summary:")
        print(summary_df.round(3).to_string())
        
        # Best model analysis
        best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
        print(f"\n🏆 Best Model: {best_model['model'].upper()}")
        print(f"  Test R²: {best_model['test_r2']:.3f}")
        print(f"  Test RMSE: {best_model['test_rmse']:.3f}")
        print(f"  Test MAE: {best_model['test_mae']:.3f}")
        print(f"  Overfitting Gap: {best_model['overfitting_gap']:.3f}")
        
        # Model recommendations
        print("\n💡 Recommendations Based on Results:")
        
        # Check for overfitting
        overfit_models = summary_df[summary_df['overfitting_gap'] > 0.15]
        if not overfit_models.empty:
            print(f"\n⚠️  Overfitting detected in: {', '.join(overfit_models['model'].values)}")
            print("  Suggestions:")
            print("  - Increase regularization parameters")
            print("  - Collect more training data")
            print("  - Use simpler models or ensemble methods")
        
        # Check for high variance
        high_variance = summary_df[summary_df['cv_mse_std'] > summary_df['cv_mse_mean'] * 0.3]
        if not high_variance.empty:
            print(f"\n⚠️  High variance in: {', '.join(high_variance['model'].values)}")
            print("  Suggestions:")
            print("  - Use more cross-validation folds")
            print("  - Apply stronger regularization")
            print("  - Consider bagging or other variance reduction techniques")
        
        # Success criteria
        good_models = summary_df[
            (summary_df['test_r2'] > 0.7) & 
            (summary_df['overfitting_gap'] < 0.15)
        ]
        if not good_models.empty:
            print(f"\n✅ Well-performing models: {', '.join(good_models['model'].values)}")
            print("  These models show good generalization")
        
        print("\n🔧 Next Steps for Improvement:")
        print("1. Try ensemble methods (Random Forest, Gradient Boosting)")
        print("2. Implement SOAP descriptors for better atomic environment representation")
        print("3. Use cross-validation for hyperparameter optimization")
        print("4. Consider neural network approaches for complex patterns")
        print("5. Collect more data if possible")
        
        print("\n✨ Analysis Complete!")
        
        return analyzer, results
        
    except FileNotFoundError:
        print("❌ Data file not found. Please run task1.py first to generate descriptors.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None