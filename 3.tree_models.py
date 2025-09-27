#!/usr/bin/env python3
"""
Robust Tree-Based Models for Au Cluster Energy Prediction
Simplified, reliable implementation focusing on core functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging

# Core dependencies (required)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# Optional dependencies (graceful degradation)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for tree-based models"""
    name: str
    model_class: Any
    param_grid: Dict
    requires_scaling: bool = False
    is_available: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    max_param_combinations: int = 27  # Reasonable limit for grid search

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass

class RobustTreeAnalyzer:
    """
    Robust Tree-Based Models Analyzer for Au Cluster Energy Prediction
    
    Features:
    - Comprehensive error handling and validation
    - Graceful degradation when optional libraries unavailable
    - Memory-efficient processing
    - Clear separation of concerns
    - Extensive logging and monitoring
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.models = {}
        self.results = {}
        self.feature_names = []
        
        # Initialize available models
        self._setup_models()
        
        logger.info(f"Initialized TreeAnalyzer with {len(self.model_configs)} available models")
    
    def _setup_models(self):
        """Setup available model configurations"""
        self.model_configs = {
            'random_forest': ModelConfig(
                name='random_forest',
                model_class=RandomForestRegressor,
                param_grid={
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2']
                },
                is_available=True
            )
        }
        
        if HAS_XGBOOST:
            self.model_configs['xgboost'] = ModelConfig(
                name='xgboost',
                model_class=xgb.XGBRegressor,
                param_grid={
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.1, 0.2],
                    'subsample': [0.8, 1.0]
                },
                is_available=True
            )
        else:
            logger.warning("XGBoost not available - skipping")
        
        if HAS_LIGHTGBM:
            self.model_configs['lightgbm'] = ModelConfig(
                name='lightgbm',
                model_class=lgb.LGBMRegressor,
                param_grid={
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.1, 0.2],
                    'num_leaves': [31, 63]
                },
                is_available=True
            )
        else:
            logger.warning("LightGBM not available - skipping")
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Comprehensive data validation"""
        logger.info("Validating input data...")
        
        # Basic shape validation
        if len(X) != len(y):
            raise DataValidationError(f"Feature matrix ({len(X)}) and target ({len(y)}) length mismatch")
        
        if len(X) < 10:
            raise DataValidationError(f"Insufficient data: {len(X)} samples (minimum 10 required)")
        
        # Missing values check
        missing_features = X.isnull().sum()
        high_missing = missing_features[missing_features > 0.5 * len(X)]
        if len(high_missing) > 0:
            logger.warning(f"Features with >50% missing values: {list(high_missing.index)}")
        
        missing_target = y.isnull().sum()
        if missing_target > 0:
            raise DataValidationError(f"Target has {missing_target} missing values")
        
        # Feature variance check
        low_variance = X.var() < 1e-10
        if low_variance.any():
            logger.warning(f"Features with near-zero variance: {list(X.columns[low_variance])}")
        
        # Infinite values check
        if np.isinf(X.values).any():
            raise DataValidationError("Features contain infinite values")
        
        if np.isinf(y.values).any():
            raise DataValidationError("Target contains infinite values")
        
        logger.info(f"Data validation passed: {X.shape[0]} samples, {X.shape[1]} features")
    
    def load_data(self, data_path: str, target_column: str = 'energy') -> pd.DataFrame:
        """Load and validate data from file path"""
        logger.info(f"Loading data from {data_path}")
        
        try:
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} rows from {data_path}")
            
            if target_column not in df.columns:
                raise DataValidationError(f"Target column '{target_column}' not found in data")
            
            return df
            
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"Data file is empty: {data_path}")
        except pd.errors.ParserError as e:
            raise DataValidationError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise DataValidationError(f"Failed to load data: {e}")
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'energy') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features with validation and cleaning"""
        logger.info("Preparing features...")
        
        if target_column not in df.columns:
            raise DataValidationError(f"Target column '{target_column}' not found in data")
        
        # Define expected feature categories
        basic_features = [
            'mean_bond_length', 'std_bond_length', 'n_bonds',
            'mean_coordination', 'std_coordination', 'max_coordination',
            'radius_of_gyration', 'asphericity', 'surface_fraction',
            'x_range', 'y_range', 'z_range', 'anisotropy',
            'compactness', 'bond_variance'
        ]
        
        # Find available features
        available_basic = [f for f in basic_features if f in df.columns]
        soap_features = [col for col in df.columns if col.startswith('soap_')]
        
        all_features = available_basic + soap_features
        
        if len(all_features) < 5:
            raise DataValidationError(f"Insufficient features available: {len(all_features)} (minimum 5 required)")
        
        logger.info(f"Using {len(available_basic)} basic + {len(soap_features)} SOAP features")
        
        # Clean data
        feature_data = df[all_features + [target_column]].copy()
        initial_rows = len(feature_data)
        feature_data = feature_data.dropna()
        final_rows = len(feature_data)
        
        if final_rows < initial_rows * 0.5:
            logger.warning(f"Dropped {initial_rows - final_rows} rows due to missing values")
        
        X = feature_data[all_features]
        y = feature_data[target_column]
        
        # Store feature names
        self.feature_names = all_features
        
        # Validate prepared data
        self.validate_data(X, y)
        
        return X, y
    
    def _limit_param_grid(self, param_grid: Dict) -> Dict:
        """Limit parameter grid size to prevent excessive computation"""
        limited_grid = {}
        total_combinations = 1
        
        for param, values in param_grid.items():
            # Limit each parameter to max 3 values
            if len(values) > 3:
                step = len(values) // 3
                limited_values = values[::step][:3]
            else:
                limited_values = values
            
            limited_grid[param] = limited_values
            total_combinations *= len(limited_values)
        
        if total_combinations > self.config.max_param_combinations:
            logger.warning(f"Parameter grid size ({total_combinations}) exceeds limit, using reduced grid")
            # Further reduce if still too large
            for param in limited_grid:
                if len(limited_grid[param]) > 2:
                    limited_grid[param] = limited_grid[param][:2]
        
        return limited_grid
    
    def train_single_model(self, config: ModelConfig, X_train: pd.DataFrame, 
                          y_train: pd.Series) -> Dict[str, Any]:
        """Train a single model with error handling"""
        logger.info(f"Training {config.name}...")
        
        try:
            # Create base model
            base_params = {
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs
            }
            
            # Add model-specific parameters
            if config.name == 'xgboost':
                base_params.update({'eval_metric': 'rmse', 'verbosity': 0})
            elif config.name == 'lightgbm':
                base_params.update({'verbose': -1})
            
            model = config.model_class(**base_params)
            
            # Limit parameter grid
            limited_params = self._limit_param_grid(config.param_grid)
            
            # Hyperparameter optimization
            if limited_params:
                grid_search = GridSearchCV(
                    model, 
                    limited_params,
                    cv=min(self.config.cv_folds, 3),  # Limit CV for speed
                    scoring='r2',
                    n_jobs=min(self.config.n_jobs, 2),  # Limit parallel jobs
                    error_score='raise'
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}
                cv_score = None
            
            return {
                'model': best_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to train {config.name}: {str(e)}")
            return {
                'model': None,
                'best_params': {},
                'cv_score': None,
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance comprehensively"""
        try:
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
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
            try:
                cv_scores = cross_val_score(
                    model, 
                    pd.concat([X_train, X_test]), 
                    pd.concat([y_train, y_test]), 
                    cv=3, 
                    scoring='r2'
                )
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                metrics['cv_mean'] = metrics['test_r2']
                metrics['cv_std'] = 0.0
            
            # Store predictions
            metrics['y_train_pred'] = y_train_pred
            metrics['y_test_pred'] = y_test_pred
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    def extract_feature_importance(self, model: Any, model_name: str) -> Optional[Dict[str, float]]:
        """Extract feature importance with error handling"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance') and callable(model.get_feature_importance):
                importances = model.get_feature_importance()
            else:
                logger.warning(f"No feature importance available for {model_name}")
                return None
            
            # Create importance dictionary
            importance_dict = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed for {model_name}: {e}")
            return None
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """Train all available models with comprehensive error handling"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        # Store splits for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        results = {}
        successful_models = 0
        
        for name, config in self.model_configs.items():
            if not config.is_available:
                logger.info(f"Skipping {name} - not available")
                continue
            
            # Train model
            training_result = self.train_single_model(config, X_train, y_train)
            
            if training_result['status'] == 'success' and training_result['model'] is not None:
                # Evaluate model
                metrics = self.evaluate_model(
                    training_result['model'], 
                    X_train, y_train, 
                    X_test, y_test
                )
                
                if metrics:  # Only proceed if evaluation succeeded
                    # Extract feature importance
                    feature_importance = self.extract_feature_importance(
                        training_result['model'], name
                    )
                    
                    # Combine all results
                    results[name] = {
                        **training_result,
                        **metrics,
                        'feature_importance': feature_importance
                    }
                    
                    successful_models += 1
                    logger.info(f"✓ {name}: R² = {metrics.get('test_r2', 0):.3f}, "
                              f"RMSE = {metrics.get('test_rmse', 0):.3f}")
                else:
                    logger.error(f"Model evaluation failed for {name}")
            else:
                logger.error(f"Model training failed for {name}: {training_result.get('error', 'Unknown error')}")
        
        if successful_models == 0:
            raise ModelTrainingError("No models trained successfully")
        
        logger.info(f"Successfully trained {successful_models}/{len(self.model_configs)} models")
        self.results = results
        return results
    
    def analyze_results(self) -> pd.DataFrame:
        """Analyze and summarize model results"""
        if not self.results:
            raise ValueError("No results to analyze - run train_all_models first")
        
        logger.info("Analyzing model results...")
        
        # Create summary dataframe
        summary_data = []
        for name, result in self.results.items():
            if result.get('status') == 'success':
                summary_data.append({
                    'model': name,
                    'train_r2': result.get('train_r2', 0),
                    'test_r2': result.get('test_r2', 0),
                    'train_rmse': result.get('train_rmse', 0),
                    'test_rmse': result.get('test_rmse', 0),
                    'cv_mean': result.get('cv_mean', 0),
                    'cv_std': result.get('cv_std', 0),
                    'overfitting': result.get('train_r2', 0) - result.get('test_r2', 0)
                })
        
        if not summary_data:
            raise ValueError("No successful model results to analyze")
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('test_r2', ascending=False)
        
        # Print analysis
        print("\n" + "="*60)
        print("TREE-BASED MODELS ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nModel Performance Ranking:")
        for i, row in summary_df.iterrows():
            print(f"{row.name + 1:2d}. {row['model'].upper():<15} | "
                  f"R² = {row['test_r2']:.3f} | "
                  f"RMSE = {row['test_rmse']:.3f} | "
                  f"CV = {row['cv_mean']:.3f}±{row['cv_std']:.3f}")
        
        # Best model analysis
        best_model = summary_df.iloc[0]
        print(f"\nBest Model: {best_model['model'].upper()}")
        print(f"  Test R²: {best_model['test_r2']:.3f}")
        print(f"  Test RMSE: {best_model['test_rmse']:.3f}")
        print(f"  Overfitting: {best_model['overfitting']:.3f}")
        
        return summary_df
    
    def create_visualizations(self, output_dir: str = './tree_results'):
        """Create essential visualizations"""
        if not self.results:
            logger.warning("No results available for visualization")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Performance comparison
        self._plot_performance_comparison(output_path)
        
        # 2. Feature importance
        self._plot_feature_importance(output_path)
        
        # 3. Predictions vs actual
        self._plot_predictions(output_path)
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_performance_comparison(self, output_path: Path):
        """Plot model performance comparison"""
        models = []
        test_r2 = []
        test_rmse = []
        cv_scores = []
        
        for name, result in self.results.items():
            if result.get('status') == 'success':
                models.append(name)
                test_r2.append(result.get('test_r2', 0))
                test_rmse.append(result.get('test_rmse', 0))
                cv_scores.append(result.get('cv_mean', 0))
        
        if not models:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² scores
        axes[0].bar(models, test_r2, alpha=0.8, color='skyblue')
        axes[0].set_ylabel('Test R² Score')
        axes[0].set_title('Model R² Performance')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # RMSE scores
        axes[1].bar(models, test_rmse, alpha=0.8, color='lightcoral')
        axes[1].set_ylabel('Test RMSE')
        axes[1].set_title('Model RMSE Performance')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Cross-validation
        axes[2].bar(models, cv_scores, alpha=0.8, color='lightgreen')
        axes[2].set_ylabel('CV R² Score')
        axes[2].set_title('Cross-Validation Performance')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, output_path: Path):
        """Plot feature importance for best model"""
        # Find best model
        best_model_name = None
        best_score = -float('inf')
        
        for name, result in self.results.items():
            if result.get('status') == 'success' and result.get('test_r2', 0) > best_score:
                best_score = result.get('test_r2', 0)
                best_model_name = name
        
        if not best_model_name or not self.results[best_model_name].get('feature_importance'):
            logger.warning("No feature importance data available")
            return
        
        importance_dict = self.results[best_model_name]['feature_importance']
        
        # Top 15 features
        top_features = list(importance_dict.items())[:15]
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importances, alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top Features - {best_model_name.title()}')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_path: Path):
        """Plot predictions vs actual for all models"""
        n_models = len([r for r in self.results.values() if r.get('status') == 'success'])
        if n_models == 0:
            return
        
        # Determine subplot layout
        cols = min(2, n_models)
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        for name, result in self.results.items():
            if result.get('status') != 'success' or 'y_test_pred' not in result:
                continue
            
            row = plot_idx // cols
            col = plot_idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            y_true = self.y_test
            y_pred = result['y_test_pred']
            
            ax.scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Add metrics
            r2 = result.get('test_r2', 0)
            rmse = result.get('test_rmse', 0)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Actual Energy (eV)')
            ax.set_ylabel('Predicted Energy (eV)')
            ax.set_title(f'{name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str = './tree_results'):
        """Save models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save models
        for name, result in self.results.items():
            if result.get('status') == 'success' and result.get('model'):
                model_path = output_path / f'{name}_model.joblib'
                try:
                    joblib.dump(result['model'], model_path)
                    logger.info(f"Saved {name} model to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save {name} model: {e}")
        
        # Save results summary
        try:
            summary_df = self.analyze_results()
            summary_df.to_csv(output_path / 'model_summary.csv', index=False)
            logger.info("Saved results summary")
        except Exception as e:
            logger.error(f"Failed to save results summary: {e}")
        
        # Save feature importance
        for name, result in self.results.items():
            importance = result.get('feature_importance')
            if importance:
                importance_df = pd.DataFrame(
                    list(importance.items()), 
                    columns=['feature', 'importance']
                )
                importance_path = output_path / f'{name}_feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"Results saved to {output_path}")

def run_tree_analysis(data_source, target_column: str = 'energy', 
                     output_dir: str = './tree_results') -> RobustTreeAnalyzer:
    """
    Main function to run robust tree analysis
    
    Parameters:
    data_source: Either a file path (str) or DataFrame with features and target
    target_column: Name of target column
    output_dir: Directory to save results
    
    Returns:
    RobustTreeAnalyzer: Trained analyzer with results
    """
    try:
        # Initialize analyzer
        analyzer = RobustTreeAnalyzer()
        
        # Handle different input types
        if isinstance(data_source, str):
            # Load from file path
            df = analyzer.load_data(data_source, target_column)
            X, y = analyzer.prepare_features(df, target_column)
        elif hasattr(data_source, 'columns'):
            # DataFrame input
            X, y = analyzer.prepare_features(data_source, target_column)
        else:
            raise ValueError("data_source must be either a file path (str) or pandas DataFrame")
        
        # Train models
        results = analyzer.train_all_models(X, y)
        
        # Analyze results
        summary_df = analyzer.analyze_results()
        
        # Create visualizations
        analyzer.create_visualizations(output_dir)
        
        # Save results
        analyzer.save_results(output_dir)
        
        print("\nTree-based analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

def main():
    """Interactive main function for command-line usage"""
    print("Robust Tree-Based Models for Au Cluster Analysis")
    print("=" * 55)
    
    try:
        # Get data path from user
        data_path = input("/Users/wilbert/Documents/GitHub/AIAC/au_cluster_analysis_results/descriptors.csv").strip()
        
        if not data_path:
            print("Using default path: ./au_cluster_analysis_results/descriptors.csv")
            data_path = "./au_cluster_analysis_results/descriptors.csv"
        
        # Get output directory
        output_dir = input("Enter output directory (press Enter for './tree_results'): ").strip()
        if not output_dir:
            output_dir = "./tree_results"
        
        # Run analysis
        analyzer = run_tree_analysis(data_path, target_column='energy', output_dir=output_dir)
        
        print("\nAnalysis completed successfully!")
        print(f"Check {output_dir} for results and visualizations")
        
        return analyzer
        
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user")
        return None
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

if __name__ == "__main__":
    # Interactive mode when run as script
    analyzer = main()
else:
    print("Robust Tree-Based Models Analyzer")
    print("Use run_tree_analysis(data_source, target_column) with your data")
    print("data_source can be either a file path or pandas DataFrame")