"""
Test Script for NaN Value Handling in Graph Neural Networks

This script loads XYZ files, processes them into graph data, and tests the NaN handling
during training. It serves as a double-check for the gradient clipping and NaN/Inf
detection mechanisms in the DualGPUGraphNeuralNetworkAnalyzer.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.optim import Adam

# Import the graph models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import reload
import importlib.util
spec = importlib.util.spec_from_file_location("graph_models", "./4.graph_models.py")
graph_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_models)

# Setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def test_nan_handling():
    """Test the NaN handling during training"""
    print("\n" + "="*60)
    print("TESTING NaN VALUE HANDLING IN GRAPH NEURAL NETWORKS")
    print("="*60)
    
    # Initialize the graph neural network analyzer
    analyzer = graph_models.DualGPUGraphNeuralNetworkAnalyzer(random_state=42)
    
    # Load XYZ files
    xyz_dir = Path('./data/Au20_OPT_1000')
    if not xyz_dir.exists():
        print(f"❌ XYZ directory {xyz_dir} not found")
        return
    
    # Load structures
    structures = analyzer._load_xyz_structures(xyz_dir)
    print(f"Loaded {len(structures)} structures")
    
    # Create graph data
    graph_data_list = []
    for structure in structures:
        atoms = structure['atoms']
        energy = structure['energy']
        if energy is not None:
            graph = analyzer._atoms_to_graph(atoms, energy)
            if graph is not None:
                graph_data_list.append(graph)
    
    print(f"Created {len(graph_data_list)} graph data objects")
    
    # Inject some problematic data (to simulate NaN occurrences)
    # This helps verify our NaN handling code works
    for i in range(min(len(graph_data_list), 3)):
        # Create extreme values in node features
        graph_data_list[i].x[0, 0] = 1e10  # Very large value
        
        # Create extreme values in edge attributes
        if graph_data_list[i].edge_attr.size(0) > 0:
            graph_data_list[i].edge_attr[0, 0] = 1e15  # Very large value
    
    print("Injected some extreme values to test NaN/Inf handling")
    
    # Split the data
    n_total = len(graph_data_list)
    n_test = max(1, int(n_total * 0.2))
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_test - n_val
    
    indices = torch.randperm(n_total)
    train_data = [graph_data_list[i] for i in indices[:n_train]]
    val_data = [graph_data_list[i] for i in indices[n_train:n_train+n_val]]
    test_data = [graph_data_list[i] for i in indices[n_train+n_val:]]
    
    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create a small SchNet model for testing
    model_name = "schnet_custom"
    config = analyzer.model_configs[model_name]
    model = analyzer.create_schnet_custom_model(config['model_params'])
    model = model.to(device)
    
    # Create data loaders
    train_loader = graph_models.DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = graph_models.DataLoader(val_data, batch_size=4, shuffle=False)
    
    # Optimizer with mixed precision training
    optimizer = Adam(model.parameters(), lr=0.001)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Custom function to check for NaN/Inf in the model
    def check_model_for_nans(model, tensor_name=""):
        has_nan = False
        has_inf = False
        for name, param in model.named_parameters():
            if param.data is not None:
                if torch.isnan(param.data).any():
                    print(f"❌ NaN detected in {tensor_name} model parameter: {name}")
                    has_nan = True
                if torch.isinf(param.data).any():
                    print(f"❌ Inf detected in {tensor_name} model parameter: {name}")
                    has_inf = True
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN detected in {tensor_name} gradient: {name}")
                    has_nan = True
                if torch.isinf(param.grad).any():
                    print(f"❌ Inf detected in {tensor_name} gradient: {name}")
                    has_inf = True
        return has_nan, has_inf
    
    # Training loop with NaN checking
    print("\nStarting training with NaN checking...")
    model.train()
    
    batch_counter = 0
    nan_detected_count = 0
    inf_detected_count = 0
    successful_updates = 0
    
    for epoch in range(2):  # Just run a couple of epochs for testing
        for batch in train_loader:
            batch_counter += 1
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Check for NaNs in input data
                if torch.isnan(batch.x).any():
                    print(f"⚠️ NaN detected in input features (batch {batch_counter})")
                    nan_detected_count += 1
                    continue
                
                if torch.isinf(batch.x).any():
                    print(f"⚠️ Inf detected in input features (batch {batch_counter})")
                    inf_detected_count += 1
                    continue
                
                # Mixed precision forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # Handle atomic numbers tensor properly
                        if batch.x.dim() > 1:
                            atomic_numbers = batch.x[:, 0].long()  # Take first column if multi-dimensional
                        else:
                            atomic_numbers = batch.x.long()
                        
                        # Forward pass
                        out = model(atomic_numbers, batch.pos, batch.batch).view(-1)
                        
                        # Check output for NaN/Inf
                        if torch.isnan(out).any():
                            print(f"⚠️ NaN detected in model output (batch {batch_counter})")
                            nan_detected_count += 1
                            continue
                        
                        if torch.isinf(out).any():
                            print(f"⚠️ Inf detected in model output (batch {batch_counter})")
                            inf_detected_count += 1
                            continue
                        
                        loss = F.mse_loss(out, batch.y)
                    
                    # Check if loss is NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        print(f"⚠️ NaN/Inf/Large loss detected: {loss.item()} (batch {batch_counter})")
                        nan_detected_count += (1 if torch.isnan(loss) else 0)
                        inf_detected_count += (1 if torch.isinf(loss) else 0)
                        continue
                    
                    # Scaled backward pass
                    scaler.scale(loss).backward()
                    
                    # FIXED: Check gradients for NaN/Inf BEFORE unscaling
                    has_inf_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            # Check for NaNs in scaled gradients
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_inf_grad = True
                                break
                    
                    if has_inf_grad:
                        print(f"⚠️ NaN/Inf detected in gradients (batch {batch_counter})")
                        nan_detected_count += 1
                        optimizer.zero_grad()
                        continue
                    
                    # Only unscale when we're sure to use the gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                
                else:
                    # Standard precision training
                    if batch.x.dim() > 1:
                        atomic_numbers = batch.x[:, 0].long()
                    else:
                        atomic_numbers = batch.x.long()
                    
                    out = model(atomic_numbers, batch.pos, batch.batch).view(-1)
                    
                    # Check output for NaN/Inf
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        print(f"⚠️ NaN/Inf detected in model output (batch {batch_counter})")
                        nan_detected_count += 1
                        continue
                    
                    loss = F.mse_loss(out, batch.y)
                    
                    # Check if loss is NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        print(f"⚠️ NaN/Inf/Large loss detected: {loss.item()} (batch {batch_counter})")
                        nan_detected_count += 1
                        continue
                    
                    loss.backward()
                    
                    # Check gradients for NaN/Inf
                    has_inf_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_inf_grad = True
                                break
                    
                    if has_inf_grad:
                        print(f"⚠️ NaN/Inf detected in gradients (batch {batch_counter})")
                        nan_detected_count += 1
                        optimizer.zero_grad()
                        continue
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                successful_updates += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ CUDA OOM, clearing cache...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"❌ Runtime error: {e}")
                    raise e
    
    print("\n" + "="*60)
    print(f"TEST RESULTS - NaN HANDLING CHECK")
    print(f"Total batches processed: {batch_counter}")
    print(f"NaN values detected: {nan_detected_count}")
    print(f"Inf values detected: {inf_detected_count}")
    print(f"Successful parameter updates: {successful_updates}")
    print(f"NaN handling ratio: {successful_updates/max(1, batch_counter):.2%}")
    print("="*60)
    
    # Final check on model parameters
    final_has_nan, final_has_inf = check_model_for_nans(model, "final")
    if not final_has_nan and not final_has_inf:
        print("✅ Model parameters are clean (no NaN/Inf values)")
        
    return {
        'total_batches': batch_counter,
        'nan_detected': nan_detected_count,
        'inf_detected': inf_detected_count,
        'successful_updates': successful_updates,
        'final_has_nan': final_has_nan,
        'final_has_inf': final_has_inf
    }

if __name__ == "__main__":
    test_nan_handling()