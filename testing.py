#!/usr/bin/env python3
"""
Enhanced Hybrid Graph Neural Network + Descriptor Training Script

Automatically detects hardware and data files, then trains:
1. Graph Neural Networks (from coordinates) - Enhanced MEGNet, CGCNN, GCN
2. Traditional ML models (from descriptors) - With better preprocessing
3. Hybrid ensemble models (combining both) - With proper scaling

Usage: python script.py
"""

import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available - no progress bars")
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

# PyTorch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    PYTORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} available")
except ImportError:
    print("ERROR: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
    print("PyTorch Geometric available")
except ImportError:
    print("ERROR: PyTorch Geometric not available. Install with: pip install torch-geometric")
    sys.exit(1)

try:
    from ase.atoms import Atoms
    ASE_AVAILABLE = True
    print("ASE available")
except ImportError:
    print("Warning: ASE not available. Install with: pip install ase")
    ASE_AVAILABLE = False


class HardwareDetector:
    """Detect and configure optimal hardware setup - SINGLE GPU ONLY"""
    
    def __init__(self):
        self.device_type = None
        self.device = None
        self.use_multi_gpu = False  # FORCED TO FALSE
        self.gpu_count = 0
        
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect best available hardware - SINGLE GPU ONLY"""
        print("\nHardware Detection (Single GPU Mode):")
        print("-" * 40)
        
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            
            print(f"CUDA GPUs detected: {self.gpu_count}")
            for i in range(self.gpu_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({memory:.1f} GB)")
            
            # FORCE SINGLE GPU USAGE
            self.device_type = "single-gpu"
            self.device = torch.device("cuda:0")
            self.use_multi_gpu = False
            print("Selected: Single GPU training (forced for stability)")
                
            torch.cuda.set_per_process_memory_fraction(0.8)  # More conservative
            
        elif torch.backends.mps.is_available():
            self.device_type = "mps"
            self.device = torch.device("mps")
            self.use_multi_gpu = False
            print("Selected: Apple Metal (MPS) acceleration")
            
        else:
            self.device_type = "cpu"
            self.device = torch.device("cpu")
            self.use_multi_gpu = False
            print("Selected: CPU training")
        
        print(f"Primary device: {self.device}")
        print("-" * 40)


class DataManager:
    """Automatically find and load both coordinate and descriptor data"""
    
    def __init__(self):
        self.coord_file = None
        self.descriptor_file = None
        self.coord_df = None
        self.descriptor_df = None
        self.merged_df = None
    
    def find_and_load_data(self):
        """Search for and load both data types"""
        print("\nSearching for data files...")
        
        # Search patterns for coordinate data
        coord_patterns = [
            "./raw_coordinates.csv",
            "./results/raw_coordinates.csv",
            "./au_cluster_analysis_results/raw_coordinates.csv",
            "./data/raw_coordinates.csv",
            "./**/raw_coordinates.csv",
            "./**/*coordinates*.csv"
        ]
        
        # Search patterns for descriptor data
        descriptor_patterns = [
            "./descriptors.csv",
            "./results/descriptors.csv", 
            "./au_cluster_analysis_results/descriptors.csv",
            "./data/descriptors.csv",
            "./**/descriptors.csv",
            "./**/*descriptor*.csv"
        ]
        
        # Find coordinate file
        for pattern in coord_patterns:
            files = list(Path(".").glob(pattern))
            for file_path in files:
                if self._validate_coordinate_file(file_path):
                    self.coord_file = str(file_path)
                    print(f"Found coordinate data: {self.coord_file}")
                    break
            if self.coord_file:
                break
        
        # Find descriptor file
        for pattern in descriptor_patterns:
            files = list(Path(".").glob(pattern))
            for file_path in files:
                if self._validate_descriptor_file(file_path):
                    self.descriptor_file = str(file_path)
                    print(f"Found descriptor data: {self.descriptor_file}")
                    break
            if self.descriptor_file:
                break
        
        # Load data
        if self.coord_file:
            self.coord_df = pd.read_csv(self.coord_file)
            print(f"Loaded coordinates: {self.coord_df.shape}")
        else:
            print("No coordinate data found")
            return False
        
        if self.descriptor_file:
            self.descriptor_df = pd.read_csv(self.descriptor_file)
            print(f"Loaded descriptors: {self.descriptor_df.shape}")
            
            # Merge datasets
            self.merged_df = self._merge_datasets()
            if self.merged_df is not None:
                print(f"Merged dataset: {self.merged_df.shape}")
                return True
        else:
            print("No descriptor data found - using coordinates only")
            return True
        
        return False
    
    def _validate_coordinate_file(self, file_path):
        """Check if file has required coordinate columns"""
        try:
            df = pd.read_csv(file_path, nrows=5)
            required_cols = ['x', 'y', 'z', 'filename', 'energy']
            return all(col in df.columns for col in required_cols)
        except:
            return False
    
    def _validate_descriptor_file(self, file_path):
        """Check if file has descriptor columns"""
        try:
            df = pd.read_csv(file_path, nrows=5)
            # Look for common descriptor patterns
            descriptor_indicators = ['mean_bond_length', 'radius_of_gyration', 'soap_pc', 'rdf_bin']
            return any(any(indicator in col for col in df.columns) for indicator in descriptor_indicators)
        except:
            return False
    
    def _merge_datasets(self):
        """Merge coordinate and descriptor datasets"""
        if self.coord_df is None or self.descriptor_df is None:
            return None
        
        # Get unique structures from coordinate data (without energy to avoid duplication)
        coord_summary = self.coord_df.groupby('filename').agg({
            'n_atoms': 'first'
        }).reset_index()
        
        # Merge with descriptors (descriptors already has energy)
        merged = coord_summary.merge(self.descriptor_df, on='filename', how='inner')
        
        print(f"Merged {len(merged)} structures with both coordinates and descriptors")
        print(f"Merged dataset columns: {list(merged.columns)}")
        
        # Verify we have energy column
        if 'energy' in merged.columns:
            print("✓ Energy column found in merged dataset")
        else:
            print("✗ No energy column in merged dataset")
            
        return merged
    
    def get_descriptor_features(self):
        """Extract clean descriptor features for ML with enhanced preprocessing"""
        if self.merged_df is None:
            return None, None
        
        print(f"Merged dataset shape: {self.merged_df.shape}")
        
        # Remove duplicate n_atoms columns (keep one)
        if 'n_atoms_x' in self.merged_df.columns and 'n_atoms_y' in self.merged_df.columns:
            # Check if they're the same
            if (self.merged_df['n_atoms_x'] == self.merged_df['n_atoms_y']).all():
                self.merged_df = self.merged_df.drop('n_atoms_y', axis=1)
                self.merged_df = self.merged_df.rename(columns={'n_atoms_x': 'n_atoms'})
        
        # Exclude non-feature columns
        exclude_cols = ['filename', 'energy', 'n_atoms', 'energy_per_atom', 'gyration_eigenvals']
        
        # Get numeric feature columns
        feature_cols = []
        for col in self.merged_df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.merged_df[col]):
                feature_cols.append(col)
        
        # Handle boolean outlier columns
        for col in feature_cols:
            if 'outlier' in col.lower() and self.merged_df[col].dtype == 'bool':
                self.merged_df[col] = self.merged_df[col].astype(int)
        
        X = self.merged_df[feature_cols].copy()
        y = self.merged_df['energy'].copy()
        
        # Remove rows with NaN/inf values
        valid_mask = ~(X.isna().any(axis=1) | y.isna() | np.isinf(X).any(axis=1) | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Feature correlation analysis and removal
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > 0.95
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        
        if high_corr_features:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        # Remove features with zero variance
        zero_var_features = X.columns[X.var() == 0].tolist()
        if zero_var_features:
            print(f"Removing {len(zero_var_features)} zero variance features")
            X = X.drop(columns=zero_var_features)
        
        print(f"Final dataset: {X.shape[1]} features, {len(X)} samples")
        print(f"Energy range: {y.min():.3f} to {y.max():.3f}")
        print(f"Feature ranges: min={X.min().min():.3f}, max={X.max().max():.3f}")
        
        return X, y


class HybridGraphNeuralNetwork:
    """Enhanced Hybrid model combining GNN and traditional ML"""
    
    def __init__(self, hardware_detector):
        self.hardware = hardware_detector
        self.device = hardware_detector.device
        self.use_multi_gpu = False  # FORCED TO FALSE
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Storage for different model types
        self.gnn_results = {}
        self.ml_results = {}
        self.hybrid_results = {}
        
        # Data scalers
        self.descriptor_scaler = StandardScaler()
        self.energy_mean = 0
        self.energy_std = 1
    
    def create_graph_data(self, coord_df):
        """Enhanced graph creation with better preprocessing"""
        print("Converting coordinates to enhanced graph format...")
        
        structures = []
        energy_values = []
        
        for filename, group in coord_df.groupby('filename'):
            try:
                coords = group[['x', 'y', 'z']].values
                elements = group['element'].values if 'element' in group.columns else ['Au'] * len(coords)
                energy = group['energy'].iloc[0]
                
                if pd.isna(energy) or len(coords) < 2:
                    continue
                
                # Filter out extreme outliers
                energy_values.append(energy)
                
                structures.append({
                    'filename': filename,
                    'energy': energy,
                    'coords': coords,
                    'elements': elements,
                    'n_atoms': len(coords)
                })
            except:
                continue
        
        # Remove energy outliers (beyond 3 std devs)
        if energy_values:
            energy_mean = np.mean(energy_values)
            energy_std = np.std(energy_values)
            energy_threshold = 3 * energy_std
            
            filtered_structures = []
            for struct in structures:
                if abs(struct['energy'] - energy_mean) <= energy_threshold:
                    filtered_structures.append(struct)
            
            print(f"Filtered {len(structures) - len(filtered_structures)} energy outliers")
            structures = filtered_structures
        
        # Convert to PyTorch Geometric graphs
        graphs = []
        iterator = tqdm(structures, desc="Creating enhanced graphs") if TQDM_AVAILABLE else structures
        
        for structure in iterator:
            try:
                graph = self._structure_to_graph(structure)
                if graph is not None and graph.x.size(0) > 1:  # Ensure valid graph
                    graphs.append(graph)
            except Exception as e:
                continue
        
        # Enhanced energy normalization
        if graphs:
            energies = [g.y.item() for g in graphs]
            
            # Use robust statistics for normalization
            energy_median = np.median(energies)
            energy_mad = np.median([abs(e - energy_median) for e in energies])
            
            self.energy_mean = energy_median
            self.energy_std = energy_mad * 1.4826  # MAD to std conversion
            
            print(f"Energy statistics: median={energy_median:.3f}, MAD={energy_mad:.3f}")
            
            for graph in graphs:
                graph.y = (graph.y - self.energy_mean) / (self.energy_std + 1e-8)
        
        print(f"Created {len(graphs)} enhanced graph objects")
        return graphs
    
    def _structure_to_graph(self, structure, cutoff=4.5):
        """Enhanced graph construction with better features"""
        coords = structure['coords']
        elements = structure['elements']
        energy = structure['energy']
        
        # Create more informative node features
        atomic_numbers = []
        for element in elements:
            if element == 'Au':
                atomic_numbers.append(79)
            elif element == 'C':
                atomic_numbers.append(6)
            elif element == 'H':
                atomic_numbers.append(1)
            else:
                atomic_numbers.append(1)
        
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # Enhanced node features
        positions = torch.tensor(coords, dtype=torch.float)
        center = positions.mean(dim=0)
        
        # 1. Atomic number one-hot encoding
        unique_atoms = torch.unique(atomic_numbers)
        if len(unique_atoms) > 1:
            min_atomic = unique_atoms.min()
            max_atomic = unique_atoms.max()
            atomic_onehot = F.one_hot(atomic_numbers - min_atomic, 
                                     num_classes=max_atomic - min_atomic + 1).float()
        else:
            atomic_onehot = torch.ones(len(atomic_numbers), 1).float()
        
        # 2. Distance to center of mass
        dist_to_center = torch.norm(positions - center, dim=1).unsqueeze(1)
        
        # 3. Coordination number (rough estimate)
        coordination = []
        for i, pos_i in enumerate(coords):
            coord_count = 0
            for j, pos_j in enumerate(coords):
                if i != j:
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < 3.0:  # Typical Au-Au bond length cutoff
                        coord_count += 1
            coordination.append(coord_count)
        
        coordination = torch.tensor(coordination, dtype=torch.float).unsqueeze(1)
        coordination = coordination / 12.0  # Normalize by max coordination
        
        # 4. Local environment features
        distances_to_others = []
        for i, pos_i in enumerate(coords):
            dists = []
            for j, pos_j in enumerate(coords):
                if i != j:
                    dists.append(np.linalg.norm(pos_i - pos_j))
            
            if dists:
                # Statistical features of local distances
                mean_dist = np.mean(sorted(dists)[:6])  # Average of 6 nearest neighbors
                std_dist = np.std(sorted(dists)[:6])
            else:
                mean_dist = std_dist = 0.0
            
            distances_to_others.append([mean_dist, std_dist])
        
        local_env = torch.tensor(distances_to_others, dtype=torch.float)
        
        # 5. Position features (normalized)
        pos_features = (positions - center) / (torch.norm(positions - center, dim=1, keepdim=True) + 1e-6)
        
        # Combine all node features
        node_features = torch.cat([
            atomic_onehot,
            dist_to_center / (dist_to_center.max() + 1e-6),  # Normalized
            coordination,
            local_env / (local_env.max() + 1e-6),  # Normalized
            pos_features
        ], dim=1)
        
        # Enhanced edge construction with multiple cutoffs
        edge_indices = []
        edge_features = []
        
        # Use adaptive cutoff based on system size
        all_distances = [
            np.linalg.norm(coords[i] - coords[j]) 
            for i in range(len(coords)) 
            for j in range(i+1, len(coords))
        ]
        
        if all_distances:
            adaptive_cutoff = min(cutoff, np.percentile(all_distances, 75))  # 75th percentile
        else:
            adaptive_cutoff = cutoff
        
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i != j:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < adaptive_cutoff:
                        edge_indices.append([i, j])
                        
                        # Enhanced edge features
                        bond_type = 1.0 if dist < 3.0 else 0.5  # Strong vs weak interaction
                        normalized_dist = dist / adaptive_cutoff
                        
                        edge_features.append([
                            dist,
                            normalized_dist,
                            bond_type,
                            1.0 / (dist + 1e-6)  # Inverse distance
                        ])
        
        if not edge_indices:
            # If no edges, create self-loops for stability
            edge_indices = [[i, i] for i in range(len(coords))]
            edge_features = [[0.1, 0.1, 1.0, 10.0] for _ in range(len(coords))]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Enhanced target: energy per atom with normalization hint
        n_atoms = len(coords)
        energy_per_atom = energy / n_atoms
        
        y = torch.tensor([energy_per_atom], dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def create_gnn_model(self, model_type, input_dim, **kwargs):
        """Create Graph Neural Network model"""
        
        if model_type == "cgcnn":
            return self._create_cgcnn_model(input_dim, **kwargs)
        elif model_type == "megnet":
            return self._create_megnet_model(input_dim, **kwargs)
        else:  # default GCN
            return self._create_gcn_model(input_dim, **kwargs)
    
    def _create_cgcnn_model(self, input_dim, hidden_dim=128, n_conv=4):
        """Create enhanced CGCNN model"""
        
        class EnhancedCGCNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_conv):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_conv = n_conv
                
                # Dynamic input embedding
                self.node_embedding = None
                
                # Enhanced convolutions with residual connections
                self.convs = nn.ModuleList([
                    GCNConv(hidden_dim, hidden_dim) for _ in range(n_conv)
                ])
                
                self.layer_norms = nn.ModuleList([
                    nn.LayerNorm(hidden_dim) for _ in range(n_conv)
                ])
                
                # Enhanced edge processing
                self.edge_embedding = nn.Sequential(
                    nn.Linear(4, hidden_dim // 2),  # 4 edge features now
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
                
                # Enhanced output layers
                self.output_layers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, 1)
                )
                
                self._input_adjusted = False
                self._init_weights()
            
            def _init_weights(self):
                """Enhanced weight initialization"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.LayerNorm):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
            
            def forward(self, data):
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                
                # Handle dynamic input size
                if self.node_embedding is None or not self._input_adjusted:
                    self.node_embedding = nn.Sequential(
                        nn.Linear(x.size(1), self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ).to(x.device)
                    self._input_adjusted = True
                
                # Node embedding
                x = self.node_embedding(x)
                x = torch.clamp(x, min=-5, max=5)
                
                # Process edge features if available
                if edge_attr is not None and edge_attr.size(1) > 0:
                    edge_features = self.edge_embedding(edge_attr)
                    edge_features = torch.clamp(edge_features, min=-5, max=5)
                
                # Enhanced convolution layers with residual connections
                for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
                    residual = x
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = F.relu(x)
                    x = torch.clamp(x, min=-5, max=5)
                    
                    # Residual connection with proper scaling
                    if i > 0:
                        x = x + 0.1 * residual
                    
                    x = F.dropout(x, p=0.1, training=self.training)
                
                # Global pooling
                x = global_mean_pool(x, batch)
                
                # Output with clamping
                output = self.output_layers(x)
                return torch.clamp(output, min=-10, max=10)
        
        return EnhancedCGCNN(input_dim, hidden_dim, n_conv)
    
    def _create_megnet_model(self, input_dim, hidden_dim=128, n_blocks=4):
        """Create enhanced MEGNet model with proper message passing"""
        
        class EnhancedMEGNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_blocks):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_blocks = n_blocks
                
                # Dynamic input handling
                self.node_embedding = None
                
                # Enhanced edge embedding
                self.edge_embedding = nn.Sequential(
                    nn.Linear(4, hidden_dim // 2),  # 4 edge features
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
                
                # True MEGNet blocks with node, edge, and global updates
                self.node_blocks = nn.ModuleList()
                self.edge_blocks = nn.ModuleList()
                self.global_blocks = nn.ModuleList()
                
                for i in range(n_blocks):
                    # Node update: node + global state
                    node_block = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                    self.node_blocks.append(node_block)
                    
                    # Edge update: edge + sender + receiver nodes
                    edge_block = nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                    self.edge_blocks.append(edge_block)
                    
                    # Global update: aggregated nodes + edges
                    global_block = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.global_blocks.append(global_block)
                
                # Learnable global state
                self.global_state = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
                
                # Enhanced output network
                self.output_layers = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),  # Combine all states
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
                
                self._input_adjusted = False
                self._init_weights()
            
            def _init_weights(self):
                """Enhanced weight initialization"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_normal_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.LayerNorm):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
            
            def forward(self, data):
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                
                # Handle dynamic input size
                if self.node_embedding is None or not self._input_adjusted:
                    self.node_embedding = nn.Sequential(
                        nn.Linear(x.size(1), self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ).to(x.device)
                    self._input_adjusted = True
                
                # Initial embeddings
                node_features = self.node_embedding(x)
                
                # Edge features
                if edge_attr is not None and edge_attr.size(1) > 0:
                    edge_features = self.edge_embedding(edge_attr)
                else:
                    edge_features = torch.zeros(edge_index.size(1), self.hidden_dim).to(x.device)
                
                # Initialize global state for each graph in batch
                batch_size = batch.max().item() + 1
                global_state = self.global_state.expand(batch_size, -1)
                
                # MEGNet message passing blocks
                for i in range(self.n_blocks):
                    # Store residuals
                    node_residual = node_features
                    edge_residual = edge_features
                    global_residual = global_state
                    
                    # Update nodes (node + global context)
                    node_global = global_state[batch]
                    node_input = torch.cat([node_features, node_global], dim=1)
                    node_features_new = self.node_blocks[i](node_input)
                    
                    # Update edges (edge + sender + receiver nodes)
                    if edge_index.size(1) > 0:
                        sender_nodes = node_features[edge_index[0]]
                        receiver_nodes = node_features[edge_index[1]]
                        edge_input = torch.cat([edge_features, sender_nodes, receiver_nodes], dim=1)
                        edge_features_new = self.edge_blocks[i](edge_input)
                    else:
                        edge_features_new = edge_features
                    
                    # Update global state
                    node_global_agg = global_mean_pool(node_features, batch)
                    if edge_index.size(1) > 0:
                        edge_batch = batch[edge_index[0]]
                        edge_global_agg = global_mean_pool(edge_features, edge_batch)
                        # Ensure correct batch size
                        if edge_global_agg.size(0) != batch_size:
                            edge_global_agg = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                    else:
                        edge_global_agg = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                    
                    global_input = torch.cat([node_global_agg, edge_global_agg], dim=1)
                    global_state_new = self.global_blocks[i](global_input)
                    
                    # Apply residual connections with gating
                    node_features = node_features_new + 0.1 * node_residual
                    edge_features = edge_features_new + 0.1 * edge_residual
                    global_state = global_state_new + 0.1 * global_residual
                    
                    # Stabilization
                    node_features = torch.clamp(node_features, min=-5, max=5)
                    edge_features = torch.clamp(edge_features, min=-5, max=5)
                    global_state = torch.clamp(global_state, min=-5, max=5)
                
                # Final aggregation
                final_node_agg = global_mean_pool(node_features, batch)
                if edge_index.size(1) > 0:
                    edge_batch = batch[edge_index[0]]
                    final_edge_agg = global_mean_pool(edge_features, edge_batch)
                    if final_edge_agg.size(0) != batch_size:
                        final_edge_agg = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                else:
                    final_edge_agg = torch.zeros(batch_size, self.hidden_dim).to(x.device)
                
                # Combine all representations
                combined = torch.cat([final_node_agg, final_edge_agg, global_state], dim=1)
                
                # Output prediction
                output = self.output_layers(combined)
                return torch.clamp(output, min=-10, max=10)
        
        return EnhancedMEGNet(input_dim, hidden_dim, n_blocks)
    
    def _create_gcn_model(self, input_dim, hidden_dim=128, n_conv=4):
        """Create enhanced GCN model"""
        
        class EnhancedGCN(nn.Module):
            def __init__(self, input_dim, hidden_dim, n_conv):
                super().__init__()
                
                self.embedding = None
                self.convs = nn.ModuleList([
                    GCNConv(hidden_dim, hidden_dim) for _ in range(n_conv)
                ])
                self.norms = nn.ModuleList([
                    nn.LayerNorm(hidden_dim) for _ in range(n_conv)
                ])
                self.dropout = nn.Dropout(0.1)
                
                self.output = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, 1)
                )
                
                self._input_dim = None
            
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                
                # Handle dynamic input size
                if self._input_dim != x.size(1):
                    self._input_dim = x.size(1)
                    self.embedding = nn.Sequential(
                        nn.Linear(x.size(1), self.convs[0].in_channels),
                        nn.LayerNorm(self.convs[0].in_channels),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ).to(x.device)
                
                x = self.embedding(x)
                
                for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                    residual = x
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = F.relu(x)
                    if i > 0:
                        x = x + 0.1 * residual
                    x = self.dropout(x)
                
                x = global_mean_pool(x, batch)
                return self.output(x)
        
        return EnhancedGCN(input_dim, hidden_dim, n_conv)
    
    def train_gnn_models(self, graph_data):
        """Train Graph Neural Network models with enhanced configurations"""
        if not graph_data:
            print("No graph data available for GNN training")
            return {}
        
        print(f"\nTraining Enhanced Graph Neural Networks on single GPU")
        
        # Split data
        n_total = len(graph_data)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        indices = torch.randperm(n_total)
        train_data = [graph_data[i] for i in indices[:n_train]]
        val_data = [graph_data[i] for i in indices[n_train:n_train+n_val]]
        test_data = [graph_data[i] for i in indices[n_train+n_val:]]
        
        print(f"GNN split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Enhanced configurations
        configs = {
            'CGCNN_Enhanced': {
                'model_type': 'cgcnn', 
                'epochs': 50, 
                'batch_size': 12, 
                'lr': 0.0005, 
                'hidden_dim': 128, 
                'n_conv': 4,
                'weight_decay': 1e-3
            },
            'MEGNet_Enhanced': {
                'model_type': 'megnet', 
                'epochs': 60, 
                'batch_size': 8, 
                'lr': 0.0003, 
                'hidden_dim': 128, 
                'n_blocks': 4,
                'weight_decay': 1e-3
            },
            'GCN_Enhanced': {
                'model_type': 'gcn', 
                'epochs': 40, 
                'batch_size': 16, 
                'lr': 0.0008, 
                'hidden_dim': 128, 
                'n_conv': 4,
                'weight_decay': 5e-4
            }
        }
        
        print("Enhanced Models to train:")
        for name, config in configs.items():
            model_type = config['model_type'].upper()
            print(f"  {name}: {model_type} - {config['epochs']} epochs, batch {config['batch_size']}")
        
        results = {}
        
        for name, config in configs.items():
            print(f"\nTraining {name} ({config['model_type'].upper()})...")
            
            try:
                # Create data loaders
                train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
                test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
                
                # Create enhanced model
                if config['model_type'] == 'cgcnn':
                    model = self.create_gnn_model('cgcnn', 10, 
                                                hidden_dim=config['hidden_dim'], 
                                                n_conv=config['n_conv'])
                elif config['model_type'] == 'megnet':
                    model = self._create_megnet_model(10,
                                                    hidden_dim=config['hidden_dim'], 
                                                    n_blocks=config['n_blocks'])
                else:  # gcn
                    model = self.create_gnn_model('gcn', 10, 
                                                hidden_dim=config['hidden_dim'], 
                                                n_conv=config['n_conv'])
                
                model = model.to(self.device)
                print(f"    Using single GPU: {self.device}")
                
                # Train with enhanced configuration
                result = self._train_single_gnn_enhanced(model, train_loader, val_loader, config, name)
                
                # Evaluate
                test_metrics = self._evaluate_gnn(result['model'], test_loader)
                
                results[name] = {
                    **result,
                    'test_r2': test_metrics['r2'],
                    'test_rmse': test_metrics['rmse'],
                    'test_mae': test_metrics['mae'],
                    'predictions': test_metrics['predictions'],
                    'targets': test_metrics['targets'],
                    'model_type': config['model_type']
                }
                
                print(f"    R² = {test_metrics['r2']:.3f}, RMSE = {test_metrics['rmse']:.3f}, Time = {result['training_time']:.1f}s")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.gnn_results = results
        
        if results:
            print(f"\nEnhanced GNN Training Summary:")
            print(f"  Successfully trained: {len(results)} models")
            best_gnn = max(results.items(), key=lambda x: x[1]['test_r2'])
            print(f"  Best GNN: {best_gnn[0]} (R² = {best_gnn[1]['test_r2']:.3f})")
        
        return results
    
    def _train_single_gnn_enhanced(self, model, train_loader, val_loader, config, name):
        """Enhanced training with better optimization"""
        # Use different optimizers for different models
        if 'MEGNet' in name:
            optimizer = AdamW(model.parameters(), lr=config['lr'], 
                             weight_decay=config['weight_decay'], betas=(0.9, 0.999))
        else:
            optimizer = Adam(model.parameters(), lr=config['lr'], 
                            weight_decay=config['weight_decay'])
        
        # More aggressive learning rate scheduling
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.7, min_lr=1e-6)
        
        best_val_loss = float('inf')
        patience = 0
        max_patience = 15
        
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        start_time = time.time()
        
        iterator = tqdm(range(config['epochs']), desc=f"Training {name}") if TQDM_AVAILABLE else range(config['epochs'])
        
        for epoch in iterator:
            # Training phase
            model.train()
            total_loss = 0
            batches = 0
            
            for batch in train_loader:
                try:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            out = model(batch).squeeze()
                            
                            # Handle dimensions
                            if out.dim() == 0:
                                out = out.unsqueeze(0)
                            if batch.y.dim() == 0:
                                batch.y = batch.y.unsqueeze(0)
                            
                            min_len = min(len(out), len(batch.y))
                            out = out[:min_len]
                            batch_y = batch.y[:min_len]
                            
                            # Add L1 regularization for sparsity
                            l1_reg = sum(p.abs().sum() for p in model.parameters())
                            loss = F.mse_loss(out, batch_y) + 1e-5 * l1_reg
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                            scaler.step(optimizer)
                            scaler.update()
                            total_loss += loss.item()
                            batches += 1
                    else:
                        out = model(batch).squeeze()
                        
                        if out.dim() == 0:
                            out = out.unsqueeze(0)
                        if batch.y.dim() == 0:
                            batch.y = batch.y.unsqueeze(0)
                        
                        min_len = min(len(out), len(batch.y))
                        out = out[:min_len]
                        batch_y = batch.y[:min_len]
                        
                        l1_reg = sum(p.abs().sum() for p in model.parameters())
                        loss = F.mse_loss(out, batch_y) + 1e-5 * l1_reg
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                            optimizer.step()
                            total_loss += loss.item()
                            batches += 1
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        batch = batch.to(self.device)
                        out = model(batch).squeeze()
                        
                        if out.dim() == 0:
                            out = out.unsqueeze(0)
                        if batch.y.dim() == 0:
                            batch.y = batch.y.unsqueeze(0)
                        
                        min_len = min(len(out), len(batch.y))
                        out = out[:min_len]
                        batch_y = batch.y[:min_len]
                        
                        loss = F.mse_loss(out, batch_y)
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            val_loss += loss.item()
                            val_batches += 1
                    except:
                        continue
            
            if batches == 0:
                print(f"No valid training batches in epoch {epoch}")
                break
            
            avg_train_loss = total_loss / batches
            avg_val_loss = val_loss / max(val_batches, 1)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
                best_state = model.state_dict().copy()
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if TQDM_AVAILABLE and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'Train': f'{avg_train_loss:.4f}',
                    'Val': f'{avg_val_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        # Load best model
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        
        return {
            'model': model,
            'training_time': time.time() - start_time,
            'best_val_loss': best_val_loss
        }
    
    def _evaluate_gnn(self, model, data_loader):
        """Evaluate GNN model - FIXED VERSION"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch = batch.to(self.device)
                    out = model(batch).squeeze()
                    
                    # Handle both single predictions and batches
                    if out.dim() == 0:
                        out = out.unsqueeze(0)
                    if batch.y.dim() == 0:
                        batch.y = batch.y.unsqueeze(0)
                    
                    # Check for valid predictions
                    valid_mask = ~(torch.isnan(out) | torch.isinf(out))
                    if valid_mask.any():
                        valid_out = out[valid_mask]
                        valid_y = batch.y[valid_mask]
                        
                        predictions.extend(valid_out.cpu().numpy())
                        targets.extend(valid_y.cpu().numpy())
                except Exception as e:
                    continue
        
        # ALWAYS return all required keys
        if not predictions:
            print("Warning: No valid predictions generated")
            return {
                'r2': 0, 
                'rmse': float('inf'), 
                'mae': float('inf'),
                'predictions': np.array([]),
                'targets': np.array([])
            }
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Denormalize
        predictions = predictions * self.energy_std + self.energy_mean
        targets = targets * self.energy_std + self.energy_mean
        
        try:
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
        except Exception as e:
            r2 = rmse = mae = 0
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def train_ml_models(self, X, y):
        """Train traditional ML models with enhanced preprocessing"""
        if X is None or len(X) == 0:
            print("No descriptor data available for ML training")
            return {}
        
        print(f"\nTraining Enhanced Traditional ML Models")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Enhanced scaling for different models
        robust_scaler = RobustScaler()
        standard_scaler = StandardScaler()
        
        X_train_robust = robust_scaler.fit_transform(X_train)
        X_val_robust = robust_scaler.transform(X_val)
        X_test_robust = robust_scaler.transform(X_test)
        
        X_train_standard = standard_scaler.fit_transform(X_train)
        X_val_standard = standard_scaler.transform(X_val)
        X_test_standard = standard_scaler.transform(X_test)
        
        print(f"ML split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        
        # Enhanced models with better regularization
        models = {
            'Random_Forest': {
                'model': RandomForestRegressor(
                    n_estimators=300, 
                    max_depth=12, 
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42, 
                    n_jobs=-1
                ),
                'data': 'raw'
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=200, 
                    max_depth=6, 
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                ),
                'data': 'raw'
            },
            'Ridge_Regression': {
                'model': Ridge(alpha=100.0),  # Much stronger regularization
                'data': 'standard'
            },
            'ElasticNet': {
                'model': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000, random_state=42),
                'data': 'robust'
            }
        }
        
        results = {}
        
        for name, config in models.items():
            print(f"\nTraining {name}...")
            
            try:
                start_time = time.time()
                model = config['model']
                
                # Select appropriate data preprocessing
                if config['data'] == 'standard':
                    X_train_use = X_train_standard
                    X_test_use = X_test_standard
                elif config['data'] == 'robust':
                    X_train_use = X_train_robust
                    X_test_use = X_test_robust
                else:  # raw
                    X_train_use = X_train
                    X_test_use = X_test
                
                # Train model
                model.fit(X_train_use, y_train)
                train_pred = model.predict(X_train_use)
                test_pred = model.predict(X_test_use)
                
                training_time = time.time() - start_time
                
                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Sanity check for R² values
                if test_r2 < -100:
                    print(f"  WARNING: Extreme negative R² ({test_r2:.2f}), model likely failed")
                    continue
                
                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'training_time': training_time,
                    'predictions': test_pred,
                    'targets': y_test.values if hasattr(y_test, 'values') else y_test,
                    'scaler': config['data']
                }
                
                print(f"  R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}, Time = {training_time:.1f}s")
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        self.ml_results = results
        return results
    
    def create_ensemble_models(self):
        """Create enhanced ensemble models with proper scaling"""
        if not self.gnn_results or not self.ml_results:
            print("Need both GNN and ML results for ensemble")
            return {}
        
        print(f"\nCreating Enhanced Ensemble Models")
        
        # Get best GNN and ML models
        best_gnn_name = max(self.gnn_results.keys(), key=lambda k: self.gnn_results[k]['test_r2'])
        
        # Filter out failed ML models (negative R² indicates failure)
        valid_ml_results = {k: v for k, v in self.ml_results.items() if v['test_r2'] > 0}
        
        if not valid_ml_results:
            print("No valid ML models for ensemble")
            return {}
        
        best_ml_name = max(valid_ml_results.keys(), key=lambda k: valid_ml_results[k]['test_r2'])
        
        best_gnn = self.gnn_results[best_gnn_name]
        best_ml = valid_ml_results[best_ml_name]
        
        print(f"Best GNN: {best_gnn_name} (R² = {best_gnn['test_r2']:.3f})")
        print(f"Best ML: {best_ml_name} (R² = {best_ml['test_r2']:.3f})")
        
        # Get predictions
        gnn_pred = best_gnn['predictions']
        ml_pred = best_ml['predictions']
        targets = best_gnn['targets']
        
        # Align predictions (take minimum length)
        min_len = min(len(gnn_pred), len(ml_pred), len(targets))
        gnn_pred = gnn_pred[:min_len]
        ml_pred = ml_pred[:min_len]
        targets = targets[:min_len]
        
        print(f"Ensemble data: {min_len} samples")
        print(f"GNN pred range: {gnn_pred.min():.3f} to {gnn_pred.max():.3f}")
        print(f"ML pred range: {ml_pred.min():.3f} to {ml_pred.max():.3f}")
        print(f"Target range: {targets.min():.3f} to {targets.max():.3f}")
        
        # Normalize predictions to same scale for ensemble
        pred_stack = np.column_stack([gnn_pred, ml_pred])
        scaler = StandardScaler()
        pred_scaled = scaler.fit_transform(pred_stack)
        
        gnn_pred_scaled = pred_scaled[:, 0]
        ml_pred_scaled = pred_scaled[:, 1]
        
        # Create ensemble predictions with scaled inputs
        ensemble_methods = {
            'Average_Ensemble': (gnn_pred_scaled + ml_pred_scaled) / 2,
            'Weighted_GNN_60': 0.6 * gnn_pred_scaled + 0.4 * ml_pred_scaled,
            'Weighted_ML_70': 0.3 * gnn_pred_scaled + 0.7 * ml_pred_scaled,
            'Weighted_ML_80': 0.2 * gnn_pred_scaled + 0.8 * ml_pred_scaled,
        }
        
        # Scale back to original range
        results = {}
        
        for name, pred_scaled in ensemble_methods.items():
            try:
                # Scale back to original target range
                pred_rescaled = pred_scaled * targets.std() + targets.mean()
                
                r2 = r2_score(targets, pred_rescaled)
                rmse = np.sqrt(mean_squared_error(targets, pred_rescaled))
                mae = mean_absolute_error(targets, pred_rescaled)
                
                results[name] = {
                    'test_r2': r2,
                    'test_rmse': rmse,
                    'test_mae': mae,
                    'predictions': pred_rescaled,
                    'targets': targets,
                    'method': name
                }
                
                print(f"  {name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
                
            except Exception as e:
                print(f"  {name} failed: {e}")
                continue
        
        # Also include best ML only (if ML is much better)
        if best_ml['test_r2'] > best_gnn['test_r2'] * 2:
            results['Best_ML_Only'] = {
                'test_r2': best_ml['test_r2'],
                'test_rmse': best_ml['test_rmse'],
                'test_mae': best_ml['test_mae'],
                'predictions': ml_pred,
                'targets': targets,
                'method': 'Best_ML_Only'
            }
            print(f"  Best_ML_Only: R² = {best_ml['test_r2']:.3f}, RMSE = {best_ml['test_rmse']:.3f}")
        
        self.hybrid_results = results
        return results
    
    def save_all_results(self):
        """Save comprehensive results from all model types"""
        output_dir = Path("./enhanced_hybrid_results")
        output_dir.mkdir(exist_ok=True)
        
        # Compile all results
        all_results = []
        
        # GNN results
        for name, result in self.gnn_results.items():
            model_type_detail = result.get('model_type', 'unknown')
            all_results.append({
                'model': name,
                'type': f'GNN_{model_type_detail.upper()}',
                'test_r2': result['test_r2'],
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae'],
                'training_time': result['training_time'],
                'device': 'single-gpu'
            })
