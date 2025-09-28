"""
Generate comprehensive model summary report for graph neural network models.
To be added to the DualGPUGraphNeuralNetworkAnalyzer class.
"""

def _generate_performance_report(self, output_dir):
    """
    Generate a comprehensive performance report for each model in CSV and text formats.
    Includes detailed metrics, training parameters, and model characteristics.
    """
    if not self.results:
        return
    
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import json
    
    # Create comprehensive summary dataframe
    report_data = []
    
    for name, result in self.results.items():
        try:
            # Get model config
            config = self.model_configs.get(name, {})
            
            # Basic metrics
            metrics = {
                'model_name': name,
                'architecture': config.get('architecture', 'unknown'),
                'test_r2': result.get('test_r2', float('nan')),
                'test_rmse': result.get('test_rmse', float('nan')),
                'test_mae': result.get('test_mae', float('nan')),
                'training_time': result.get('training_time', float('nan')),
                'gpu_device': result.get('gpu_device', 'unknown'),
                'epochs_trained': len(result.get('train_losses', [])),
                'early_stopping': result.get('early_stopping', False)
            }
            
            # Get more detailed metrics
            train_losses = result.get('train_losses', [])
            val_losses = result.get('val_losses', [])
            
            if train_losses and val_losses:
                # Find best epoch
                best_epoch = np.argmin(val_losses) + 1
                best_val_loss = min(val_losses)
                final_train_loss = train_losses[-1]
                final_val_loss = val_losses[-1]
                
                # Overfitting indicators
                train_val_ratio = final_val_loss / (final_train_loss + 1e-10)
                loss_decrease_rate = 0
                if len(train_losses) > 5:
                    # Calculate average loss decrease rate over last 5 epochs
                    loss_decrease_rate = np.mean(np.diff(train_losses[-6:-1]))
                
                # Add to metrics
                metrics.update({
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val_loss,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'train_val_ratio': train_val_ratio,
                    'loss_decrease_rate': loss_decrease_rate,
                    'convergence': 'Yes' if abs(loss_decrease_rate) < 1e-4 else 'No'
                })
            
            # Extract model parameters
            model_params = config.get('model_params', {})
            training_params = config.get('training_params', {})
            
            # Add important model parameters
            if model_params:
                for key in ['hidden_channels', 'num_filters', 'num_interactions', 'num_gaussians', 'cutoff']:
                    if key in model_params:
                        metrics[f'param_{key}'] = model_params[key]
            
            # Add important training parameters
            if training_params:
                for key in ['batch_size', 'learning_rate', 'weight_decay']:
                    if key in training_params:
                        metrics[f'train_{key}'] = training_params[key]
            
            # Calculate predictions and residuals metrics
            if 'predictions' in result and 'targets' in result:
                predictions = np.array(result['predictions'])
                targets = np.array(result['targets'])
                residuals = targets - predictions
                
                metrics.update({
                    'residuals_mean': np.mean(residuals),
                    'residuals_std': np.std(residuals),
                    'residuals_skew': 0 if len(residuals) == 0 else float(pd.Series(residuals).skew()),
                    'predictions_min': np.min(predictions),
                    'predictions_max': np.max(predictions),
                    'targets_min': np.min(targets),
                    'targets_max': np.max(targets)
                })
            
            # Add to report data
            report_data.append(metrics)
        
        except Exception as e:
            print(f"Error generating report for {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create DataFrame from report data
    if report_data:
        report_df = pd.DataFrame(report_data)
        
        # Save to CSV
        report_df.to_csv(output_dir / 'model_summary.csv', index=False)
        
        # Generate text report
        with open(output_dir / 'model_performance_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRAPH NEURAL NETWORK MODELS PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Performance ranking
            f.write("MODEL PERFORMANCE RANKING (by R²)\n")
            f.write("-" * 50 + "\n")
            
            for i, (_, row) in enumerate(report_df.sort_values('test_r2', ascending=False).iterrows()):
                f.write(f"{i+1}. {row['model_name']} ({row['architecture']})\n")
                f.write(f"   R²: {row['test_r2']:.4f}, RMSE: {row['test_rmse']:.4f}, MAE: {row['test_mae']:.4f}\n")
                f.write(f"   Training time: {row['training_time']:.2f}s on {row['gpu_device']}\n")
                f.write("\n")
            
            # Detailed model analysis
            f.write("\nDETAILED MODEL ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for _, row in report_df.iterrows():
                f.write(f"MODEL: {row['model_name'].upper()} ({row['architecture']})\n")
                f.write("-" * 50 + "\n")
                f.write(f"Performance Metrics:\n")
                f.write(f"  - R² Score: {row['test_r2']:.4f}\n")
                f.write(f"  - RMSE: {row['test_rmse']:.4f}\n")
                f.write(f"  - MAE: {row['test_mae']:.4f}\n")
                f.write(f"  - Training Device: {row['gpu_device']}\n")
                f.write(f"  - Training Time: {row['training_time']:.2f} seconds\n")
                
                if 'best_epoch' in row:
                    f.write(f"\nTraining Dynamics:\n")
                    f.write(f"  - Total Epochs: {row['epochs_trained']}\n")
                    f.write(f"  - Best Epoch: {row['best_epoch']}\n")
                    f.write(f"  - Early Stopping: {'Yes' if row['early_stopping'] else 'No'}\n")
                    f.write(f"  - Best Validation Loss: {row['best_val_loss']:.4f}\n")
                    f.write(f"  - Final Train/Val Loss Ratio: {row['train_val_ratio']:.4f}")
                    f.write(f" {'(Potential overfitting)' if row['train_val_ratio'] > 1.2 else ''}\n")
                    f.write(f"  - Convergence Status: {row['convergence']}\n")
                
                f.write("\nModel Parameters:\n")
                param_cols = [c for c in row.index if c.startswith('param_')]
                for col in param_cols:
                    param_name = col.replace('param_', '')
                    f.write(f"  - {param_name}: {row[col]}\n")
                
                f.write("\nTraining Parameters:\n")
                train_cols = [c for c in row.index if c.startswith('train_')]
                for col in train_cols:
                    param_name = col.replace('train_', '')
                    f.write(f"  - {param_name}: {row[col]}\n")
                
                # Add residual analysis if available
                if 'residuals_mean' in row:
                    f.write("\nResidual Analysis:\n")
                    f.write(f"  - Mean: {row['residuals_mean']:.4f}\n")
                    f.write(f"  - Std Dev: {row['residuals_std']:.4f}\n")
                    f.write(f"  - Skewness: {row['residuals_skew']:.4f}\n")
                    
                    # Interpretation
                    f.write("\nInterpretation:\n")
                    if abs(row['residuals_mean']) < 0.1 * row['residuals_std']:
                        f.write("  - Residuals are centered near zero (unbiased prediction)\n")
                    else:
                        f.write("  - Residuals show potential bias\n")
                    
                    if abs(row['residuals_skew']) < 0.5:
                        f.write("  - Residuals are approximately normally distributed\n")
                    else:
                        skew_dir = "right" if row['residuals_skew'] > 0 else "left"
                        f.write(f"  - Residuals are skewed to the {skew_dir}\n")
                
                # Add prediction range analysis
                if 'predictions_min' in row:
                    target_range = row['targets_max'] - row['targets_min']
                    pred_range = row['predictions_max'] - row['predictions_min']
                    range_ratio = pred_range / target_range if target_range > 0 else 0
                    
                    f.write("\nPrediction Range Analysis:\n")
                    f.write(f"  - Target range: [{row['targets_min']:.4f}, {row['targets_max']:.4f}]\n")
                    f.write(f"  - Prediction range: [{row['predictions_min']:.4f}, {row['predictions_max']:.4f}]\n")
                    f.write(f"  - Range coverage: {range_ratio:.2%}\n")
                    
                    if range_ratio < 0.8:
                        f.write("  - WARNING: Model predictions have compressed range compared to targets\n")
                    elif range_ratio > 1.2:
                        f.write("  - WARNING: Model predictions have expanded range compared to targets\n")
                    else:
                        f.write("  - Model prediction range matches target range well\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
                
            # Overall summary
            f.write("\nOVERALL SUMMARY\n")
            f.write("-" * 50 + "\n")
            best_model = report_df.loc[report_df['test_r2'].idxmax()]
            fastest_model = report_df.loc[report_df['training_time'].idxmin()]
            
            f.write(f"Best performing model: {best_model['model_name']} (R²: {best_model['test_r2']:.4f})\n")
            f.write(f"Fastest model: {fastest_model['model_name']} ({fastest_model['training_time']:.2f}s)\n")
            
            if self.dual_gpu_mode:
                gpu0_models = report_df[report_df['gpu_device'] == 'cuda:0']
                gpu1_models = report_df[report_df['gpu_device'] == 'cuda:1']
                
                if not gpu0_models.empty and not gpu1_models.empty:
                    gpu0_time = gpu0_models['training_time'].sum()
                    gpu1_time = gpu1_models['training_time'].sum()
                    sequential_time = report_df['training_time'].sum()
                    parallel_time = max(gpu0_time, gpu1_time)
                    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
                    
                    f.write(f"\nDual GPU Performance:\n")
                    f.write(f"  - Sequential training estimate: {sequential_time:.2f}s\n")
                    f.write(f"  - Actual parallel training time: {parallel_time:.2f}s\n")
                    f.write(f"  - Speedup achieved: {speedup:.2f}x\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
        print(f"📊 Comprehensive performance report saved to {output_dir / 'model_performance_report.txt'}")
        
        # Also save results as JSON for easier programmatic access
        try:
            # Convert NumPy values to Python native types for JSON serialization
            json_report = report_df.to_dict(orient='records')
            for record in json_report:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        record[key] = value.tolist()
                    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        record[key] = str(value)
                    elif pd.isna(value):
                        record[key] = None
            
            with open(output_dir / 'model_performance_summary.json', 'w') as f:
                json.dump(json_report, f, indent=2)
                
        except Exception as e:
            print(f"Error saving JSON report: {e}")
    
    else:
        print("No report data available.")