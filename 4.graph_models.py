import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
import os
warnings.filterwarnings('ignore')

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️  tqdm not available - no progress bars")
    TQDM_AVAILABLE = False
    tqdm = None

# PyTorch and PyTorch Geometric with Distributed Support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SchNet, DimeNet, GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.utils import from_networkx
    from torch.nn import LayerNorm, BatchNorm1d
    
    # Distributed training imports
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    
    # Enhanced device detection with distributed support
    available_gpus = []
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            available_gpus.append(f"cuda:{i}")
        
        print(f"✅ {gpu_count} CUDA GPU(s) available:")
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        
        # Set primary device
        device = torch.device("cuda:0")
        print(f"Primary device: cuda:0")
        
        # Set memory management
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory max
        
        if gpu_count >= 2:
            print("🚀 Multi-GPU training with DistributedDataParallel will be enabled")
            MULTI_GPU_AVAILABLE = True
        else:
            print("⚠️  Single GPU detected - using DataParallel if applicable")
            MULTI_GPU_AVAILABLE = False
            
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        available_gpus = ["mps"]
        MULTI_GPU_AVAILABLE = False
        print("✅ PyTorch MPS (Metal) acceleration available")
    else:
        device = torch.device("cpu")
        available_gpus = ["cpu"]
        MULTI_GPU_AVAILABLE = False
        print("⚠️  Using CPU (CUDA not available)")
    
    PYTORCH_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ PyTorch Geometric not available: {e}")
    PYTORCH_AVAILABLE = False
    MULTI_GPU_AVAILABLE = False
    device = None
    available_gpus = []
    torch = None
    nn = None
    F = None

# ASE for molecular structures
try:
    from ase.atoms import Atoms
    from ase.neighborlist import NeighborList, natural_cutoffs
    ASE_AVAILABLE = True
except ImportError:
    print("❌ ASE not available")
    ASE_AVAILABLE = False


class DistributedGraphNeuralNetworkAnalyzer:
    """
    Graph Neural Network Models with Distributed Data Parallel Support
    """
    
    def __init__(self, random_state=42, use_distributed=True):
        self.random_state = random_state
        self.use_distributed = use_distributed and MULTI_GPU_AVAILABLE
        
        if PYTORCH_AVAILABLE:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.manual_seed_all(random_state)
        
        self.models = {}
        self.results = {}
        self.device = device
        self.available_gpus = available_gpus
        self.world_size = len(available_gpus) if torch.cuda.is_available() else 1
        self.rank = 0  # Will be set during distributed training
        
        # Initialize distributed training if available
        self.is_distributed = False
        if self.use_distributed and torch.cuda.is_available():
            self._setup_distributed()
        
        # Graph neural network architectures
        self.model_configs = self._initialize_models()
        
        if self.is_distributed:
            print(f"🚀 DISTRIBUTED MODE: Using {self.world_size} GPUs with DistributedDataParallel")
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"📊 DATAPARALLEL MODE: Using {torch.cuda.device_count()} GPUs with DataParallel")
            print("💡 For true distributed training, use: torchrun --nproc_per_node=2 script.py")
        else:
            print(f"🔧 SINGLE DEVICE MODE: Using {self.device}")
        
        print("🎯 ENHANCED MODE: Automatic multi-GPU support with fallbacks")
    
    def _setup_distributed(self):
        """Setup distributed training environment"""
        try:
            # Check if we're in a REAL distributed environment (launched with torchrun/mpirun)
            if ('RANK' in os.environ and 'WORLD_SIZE' in os.environ and 
                'LOCAL_RANK' in os.environ and 'MASTER_ADDR' in os.environ):
                
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                local_rank = int(os.environ['LOCAL_RANK'])
                
                # Initialize the process group
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
                
                # Set the device for this process
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f'cuda:{local_rank}')
                self.is_distributed = True
                
                print(f"Distributed training initialized: rank {self.rank}/{self.world_size} on {self.device}")
                
            else:
                # Not in a distributed environment - fall back to DataParallel
                print("Not in distributed environment, falling back to DataParallel mode")
                self.is_distributed = False
                self.use_distributed = False
                
        except Exception as e:
            print(f"Failed to setup distributed training: {e}")
            print("Falling back to DataParallel mode")
            self.is_distributed = False
            self.use_distributed = False
    
    def _cleanup_distributed(self):
        """Cleanup distributed training"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache for all available devices"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _initialize_models(self):
        """Initialize graph neural network architectures with distributed-friendly parameters"""
        if not PYTORCH_AVAILABLE:
            return {}
        
        # Adjust parameters based on available resources
        if self.use_distributed and self.world_size > 1:
            # Distributed training - can use larger batch sizes and more epochs
            schnet_epochs, schnet_batch = 80, 64 * self.world_size  # Scale batch size
            cgcnn_epochs, cgcnn_batch = 150, 32 * self.world_size
            megnet_epochs, megnet_batch = 180, 32 * self.world_size
            learning_rate = 0.001 * np.sqrt(self.world_size)  # Scale learning rate
            weight_decay = 1e-5
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Multi-GPU DataParallel
            schnet_epochs, schnet_batch = 70, 128
            cgcnn_epochs, cgcnn_batch = 130, 64
            megnet_epochs, megnet_batch = 160, 64
            learning_rate = 0.001
            weight_decay = 1e-5
        elif torch.cuda.is_available():
            # Single GPU
            schnet_epochs, schnet_batch = 60, 64
            cgcnn_epochs, cgcnn_batch = 120, 32
            megnet_epochs, megnet_batch = 150, 32
            learning_rate = 0.001
            weight_decay = 1e-4
        else:
            # CPU training
            schnet_epochs, schnet_batch = 40, 16
            cgcnn_epochs, cgcnn_batch = 80, 16
            megnet_epochs, megnet_batch = 100, 16
            learning_rate = 0.0005
            weight_decay = 1e-4
        
        configs = {
            'schnet_custom': {
                'architecture': 'schnet_custom',
                'model_params': {
                    'hidden_channels': 128,
                    'num_filters': 128,
                    'num_interactions': 6,
                    'num_gaussians': 50,
                    'cutoff': 6.0,
                    'max_num_neighbors': 32
                },
                'training_params': {
                    'learning_rate': learning_rate,
                    'batch_size': schnet_batch,
                    'epochs': schnet_epochs,
                    'weight_decay': weight_decay,
                    'optimizer': 'adamw',
                    'scheduler': 'cosine'
                }
            },
            'cgcnn': {
                'architecture': 'cgcnn',
                'model_params': {
                    'atom_fea_len': 64,
                    'h_fea_len': 128,
                    'n_conv': 4,
                    'n_h': 2
                },
                'training_params': {
                    'learning_rate': learning_rate,
                    'batch_size': cgcnn_batch,
                    'epochs': cgcnn_epochs,
                    'weight_decay': weight_decay,
                    'optimizer': 'adamw',
                    'scheduler': 'reduce_on_plateau'
                }
            },
            'megnet': {
                'architecture': 'megnet_inspired',
                'model_params': {
                    'node_dim': 64,
                    'edge_dim': 32,
                    'global_dim': 32,
                    'hidden_dim': 128,
                    'n_blocks': 3
                },
                'training_params': {
                    'learning_rate': learning_rate,
                    'batch_size': megnet_batch,
                    'epochs': megnet_epochs,
                    'weight_decay': weight_decay,
                    'optimizer': 'adam',
                    'scheduler': 'cosine'
                }
            }
        }
        
        return configs
    
    def load_data(self, data_path=None, structures_data=None, xyz_dir=None, target_column='energy'):
        """Load data and convert to graph format"""
        if data_path and structures_data:
            if isinstance(data_path, str):
                self.df = pd.read_csv(data_path)
            else:
                self.df = data_path
            self.structures_data = structures_data
        elif xyz_dir:
            print(f"Loading structures directly from XYZ files in: {xyz_dir}")
            self.structures_data = self._load_xyz_structures(xyz_dir)
            self.df = self._create_dataframe_from_structures()
        elif data_path:
            if isinstance(data_path, str):
                self.df = pd.read_csv(data_path)
            else:
                self.df = data_path
            
            coord_cols = ['x', 'y', 'z']
            if all(col in self.df.columns for col in coord_cols):
                print("CSV contains coordinate data - reconstructing molecular structures")
                self.structures_data = self._reconstruct_structures_from_csv()
            else:
                print("CSV does not contain coordinate data - trying to find XYZ files")
                csv_path = Path(data_path) if isinstance(data_path, str) else Path(".")
                possible_xyz_dir = csv_path.parent.parent / "data" / "Au20_OPT_1000"
                if possible_xyz_dir.exists():
                    print(f"Found XYZ directory at: {possible_xyz_dir}")
                    self.structures_data = self._load_xyz_structures(possible_xyz_dir)
                else:
                    print("No coordinate data available. Graph models need atomic coordinates.")
                    self.structures_data = None
        else:
            raise ValueError("Must provide either data_path with structures_data, or xyz_dir, or data_path with coordinates")
        
        if target_column in self.df.columns:
            if hasattr(self, '_structures_df'):
                self._structures_df = self._structures_df.dropna(subset=[target_column])
                
                # Data quality checks
                energy_values = self._structures_df[target_column]
                outliers = energy_values < (energy_values.quantile(0.01))
                outliers |= energy_values > (energy_values.quantile(0.99))
                
                if outliers.sum() > 0:
                    print(f"⚠️  Removing {outliers.sum()} outlier structures")
                    self._structures_df = self._structures_df[~outliers]
                
                print(f"Loaded {len(self._structures_df)} unique structures for graph neural networks")
                print(f"Target range: {self._structures_df[target_column].min():.2f} to {self._structures_df[target_column].max():.2f}")
                print(f"Target std: {self._structures_df[target_column].std():.2f}")
            else:
                self.df = self.df.dropna(subset=[target_column])
                
                # Data quality checks for regular df
                energy_values = self.df[target_column]
                outliers = energy_values < (energy_values.quantile(0.01))
                outliers |= energy_values > (energy_values.quantile(0.99))
                
                if outliers.sum() > 0:
                    print(f"⚠️  Removing {outliers.sum()} outlier samples")
                    self.df = self.df[~outliers]
                
                print(f"Loaded {len(self.df)} samples for graph neural networks")
                print(f"Target range: {self.df[target_column].min():.2f} to {self.df[target_column].max():.2f}")
                print(f"Target std: {self.df[target_column].std():.2f}")
        
        return self.df
    
    def _reconstruct_structures_from_csv(self):
        """Reconstruct molecular structures from CSV file with coordinates"""
        print("Reconstructing molecular structures from CSV coordinate data...")
        
        structures = []
        grouped = self.df.groupby('filename')
        structure_data = []
        
        for filename, group in grouped:
            try:
                coords = group[['x', 'y', 'z']].values
                elements = group['element'].values if 'element' in group.columns else ['Au'] * len(coords)
                energy = group['energy'].iloc[0] if 'energy' in group.columns else None
                n_atoms = len(coords)
                
                if ASE_AVAILABLE:
                    atoms = Atoms(symbols=elements, positions=coords)
                else:
                    atoms = None
                
                structure = {
                    'filename': filename,
                    'n_atoms': n_atoms,
                    'energy': energy,
                    'atoms': atoms,
                    'coords': coords,
                    'elements': elements
                }
                
                structures.append(structure)
                structure_data.append({
                    'filename': filename,
                    'n_atoms': n_atoms,
                    'energy': energy,
                    'energy_per_atom': energy / n_atoms if energy else None
                })
                
            except Exception as e:
                print(f"Error reconstructing structure {filename}: {e}")
                continue
        
        self._structures_df = pd.DataFrame(structure_data)
        print(f"Successfully reconstructed {len(structures)} molecular structures from CSV")
        return structures
    
    def _load_xyz_structures(self, xyz_dir):
        """Load structures directly from XYZ files"""
        xyz_dir = Path(xyz_dir)
        xyz_files = list(xyz_dir.glob("*.xyz"))
        
        print(f"Found {len(xyz_files)} XYZ files")
        
        structures = []
        
        # Create progress bar for XYZ file loading
        if TQDM_AVAILABLE:
            xyz_iter = tqdm(xyz_files, desc="Loading XYZ files", unit="file")
        else:
            xyz_iter = xyz_files
        
        for xyz_file in xyz_iter:
            try:
                coords = []
                elements = []
                energy = None
                
                with open(xyz_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    continue
                
                n_atoms = int(lines[0].strip())
                energy_line = lines[1].strip()
                numbers = []
                import re
                for match in re.finditer(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', energy_line):
                    try:
                        val = float(match.group())
                        if -50000 < val < 50000:
                            numbers.append(val)
                    except:
                        continue
                
                if numbers:
                    energy = numbers[0]
                
                for i in range(2, min(len(lines), n_atoms + 2)):
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        element = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        elements.append(element)
                        coords.append([x, y, z])
                
                if len(coords) > 0:
                    atoms = Atoms(symbols=elements, positions=coords)
                    
                    structures.append({
                        'filename': xyz_file.name,
                        'n_atoms': len(atoms),
                        'energy': energy,
                        'atoms': atoms,
                        'coords': np.array(coords),
                        'elements': elements
                    })
                
            except Exception as e:
                print(f"Error parsing {xyz_file.name}: {e}")
                continue
        
        print(f"Successfully parsed {len(structures)} XYZ files")
        return structures
    
    def _create_dataframe_from_structures(self):
        """Create DataFrame from loaded structures"""
        data = []
        for structure in self.structures_data:
            data.append({
                'filename': structure['filename'],
                'n_atoms': structure['n_atoms'],
                'energy': structure['energy'],
                'energy_per_atom': structure['energy'] / structure['n_atoms'] if structure['energy'] else None
            })
        
        return pd.DataFrame(data)
    
    def create_graph_data(self, target_column='energy'):
        """Convert molecular structures to PyTorch Geometric Data objects"""
        if not PYTORCH_AVAILABLE or not ASE_AVAILABLE:
            print("❌ PyTorch Geometric or ASE not available")
            return None
        
        if self.structures_data is None:
            print("❌ No structures data provided")
            return None
        
        print("Converting molecular structures to graph format...")
        
        graph_data_list = []
        
        # Create progress bar for structure conversion
        if TQDM_AVAILABLE:
            structure_iter = tqdm(self.structures_data, desc="Converting to graphs", unit="structure")
        else:
            structure_iter = self.structures_data
        
        for structure in structure_iter:
            try:
                atoms = structure['atoms'] if 'atoms' in structure else None
                if atoms is None and ASE_AVAILABLE:
                    coords = structure['coords']
                    elements = structure.get('elements', ['Au'] * len(coords))
                    atoms = Atoms(symbols=elements, positions=coords)
                elif atoms is None:
                    print("Cannot create Atoms object - ASE not available")
                    continue
                
                energy = structure['energy']
                if energy is None or pd.isna(energy):
                    continue
                
                graph = self._atoms_to_graph(atoms, energy)
                if graph is not None:
                    graph_data_list.append(graph)
                
            except Exception as e:
                print(f"Error converting {structure.get('filename', 'unknown')} to graph: {e}")
                continue
        
        print(f"Successfully converted {len(graph_data_list)} structures to graphs")
        
        # Normalize energy values for better training
        if len(graph_data_list) > 0:
            energies = [data.y.item() for data in graph_data_list]
            self.energy_mean = np.mean(energies)
            self.energy_std = np.std(energies)
            
            print(f"Energy statistics: Mean = {self.energy_mean:.3f}, Std = {self.energy_std:.3f}")
            
            # Normalize energies
            for data in graph_data_list:
                data.y = (data.y - self.energy_mean) / (self.energy_std + 1e-8)
        
        return graph_data_list
    
    def _atoms_to_graph(self, atoms, energy, cutoff=5.0):
        """Convert ASE Atoms object to PyTorch Geometric Data with improved features"""
        n_atoms = len(atoms)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float, requires_grad=False)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        
        # Richer node features
        unique_elements = torch.unique(atomic_numbers)
        max_atomic_num = atomic_numbers.max().item()
        min_atomic_num = atomic_numbers.min().item()
        node_features = F.one_hot(atomic_numbers - min_atomic_num, 
                                 num_classes=max_atomic_num - min_atomic_num + 1).float()
        
        # Add positional features
        center = positions.mean(dim=0)
        distances_from_center = torch.norm(positions - center, dim=1).unsqueeze(1)
        
        # Add coordination number feature
        coord_numbers = []
        for i in range(n_atoms):
            coord_count = 0
            for j in range(n_atoms):
                if i != j and torch.norm(positions[i] - positions[j]) < cutoff:
                    coord_count += 1
            coord_numbers.append([coord_count])
        coord_features = torch.tensor(coord_numbers, dtype=torch.float)
        
        # Combine all node features and ensure gradient tracking is enabled
        node_features = torch.cat([node_features, distances_from_center, coord_features], dim=1)
        node_features = node_features.detach().requires_grad_(True)  # FIXED: Explicitly enable gradients
        
        edge_indices = []
        edge_features = []
        
        # Better edge features
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    distance = torch.norm(positions[i] - positions[j])
                    if distance < cutoff:
                        edge_indices.append([i, j])
                        rel_pos = positions[j] - positions[i]
                        edge_features.append([
                            distance.item(),
                            rel_pos[0].item(),
                            rel_pos[1].item(), 
                            rel_pos[2].item(),
                            1.0 / (distance.item() + 1e-6)  # Inverse distance
                        ])
        
        if len(edge_indices) == 0:
            return None
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float, requires_grad=False)
        
        # Normalize energy for better training
        y = torch.tensor([energy / n_atoms], dtype=torch.float, requires_grad=False)  # Energy per atom
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            y=y
        )
        
        return data
    
    def create_schnet_custom_model(self, model_params):
        """Create custom SchNet model"""
        class CustomSchNet(nn.Module):
            def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6, 
                         num_gaussians=50, cutoff=5.0, max_num_neighbors=32):
                super().__init__()
                self.hidden_channels = hidden_channels
                self.num_filters = num_filters
                self.num_interactions = num_interactions
                self.cutoff = cutoff
                self.max_num_neighbors = max_num_neighbors
                
                # FIXED: Use linear layer instead of embedding for node features
                self.node_embedding = None  # Will be created dynamically
                self.distance_expansion = nn.Sequential(
                    nn.Linear(1, num_gaussians),
                    nn.ReLU()
                )
                
                self.interactions = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_channels + num_gaussians, num_filters),
                        nn.ReLU(),
                        nn.Linear(num_filters, hidden_channels),
                        nn.ReLU()
                    ) for _ in range(num_interactions)
                ])
                
                self.output_layers = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels // 2, 1)
                )
                
                self._input_adjusted = False
            
            def _build_edges_simple(self, pos, batch):
                """Simple edge construction"""
                batch_size = batch.max().item() + 1
                edge_indices = []
                edge_distances = []
                
                for b in range(batch_size):
                    mask = batch == b
                    if mask.sum() < 2:
                        continue
                    
                    batch_pos = pos[mask]
                    n_atoms = batch_pos.size(0)
                    
                    for i in range(n_atoms):
                        for j in range(n_atoms):
                            if i != j:
                                dist = torch.norm(batch_pos[i] - batch_pos[j])
                                if dist <= self.cutoff:
                                    global_i = torch.where(mask)[0][i]
                                    global_j = torch.where(mask)[0][j]
                                    edge_indices.append([global_i.item(), global_j.item()])
                                    edge_distances.append(dist.item())
                
                if len(edge_indices) == 0:
                    return torch.zeros(2, 0, dtype=torch.long, device=pos.device), torch.zeros(0, device=pos.device)
                
                edge_index = torch.tensor(edge_indices, dtype=torch.long, device=pos.device).t()
                edge_distances = torch.tensor(edge_distances, device=pos.device)
                
                return edge_index, edge_distances
            
            def forward(self, x, pos, batch):
                """
                Forward pass using node features directly instead of atomic numbers
                Args:
                    x: Node features tensor (batch_size, feature_dim)  
                    pos: Node positions (batch_size, 3)
                    batch: Batch indices
                """
                # FIXED: Use node features directly instead of atomic numbers
                # Handle dynamic input size for node features
                if self.node_embedding is None or not self._input_adjusted:
                    input_dim = x.size(1)
                    self.node_embedding = nn.Linear(input_dim, self.hidden_channels).to(x.device)
                    self._input_adjusted = True
                
                # Embed node features to hidden dimensions
                h = self.node_embedding(x.float())
                
                edge_index, edge_distances = self._build_edges_simple(pos, batch)
                
                if edge_index.size(1) == 0:
                    return global_mean_pool(h, batch)
                
                edge_distances = edge_distances.unsqueeze(-1)
                edge_features = self.distance_expansion(edge_distances)
                
                h = h.float()
                edge_features = edge_features.float()
                
                for interaction in self.interactions:
                    row, col = edge_index
                    node_features = h[row]
                    
                    if node_features.dim() == 3:
                        node_features = node_features.squeeze(1)
                    
                    edge_input = torch.cat([node_features, edge_features], dim=-1)
                    edge_update = interaction(edge_input)
                    
                    edge_update = edge_update.float()
                    h_new = h.clone().float()
                    h_new.index_add_(0, col, edge_update)
                    h = h_new
                
                graph_features = global_mean_pool(h, batch)
                return self.output_layers(graph_features)
        
        return CustomSchNet(**model_params)
    
    def create_cgcnn_model(self, model_params):
        """Create CGCNN model with improved architecture and proper gradient handling"""
        class CGCNNModel(nn.Module):
            def __init__(self, atom_fea_len=64, h_fea_len=128, n_conv=3, n_h=1):
                super().__init__()
                self.atom_fea_len = atom_fea_len
                self.h_fea_len = h_fea_len
                self.n_conv = n_conv
                
                # FIXED: Handle variable input feature size properly
                self.embedding = nn.Linear(atom_fea_len, self.atom_fea_len)
                self.batch_norm_input = nn.BatchNorm1d(self.atom_fea_len, track_running_stats=False)
                
                self.convs = nn.ModuleList([
                    GCNConv(self.atom_fea_len if i == 0 else self.h_fea_len, self.h_fea_len)
                    for i in range(self.n_conv)
                ])
                
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(self.h_fea_len, track_running_stats=False) for _ in range(self.n_conv)
                ])
                
                self.fc = nn.Sequential(
                    nn.Linear(self.h_fea_len, self.h_fea_len),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.h_fea_len, track_running_stats=False),
                    nn.Dropout(0.3),
                    nn.Linear(self.h_fea_len, self.h_fea_len // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.h_fea_len // 2, 1)
                )
                
                # FIXED: Track input feature size adjustment
                self._input_adjusted = False
            
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                
                # FIXED: Ensure input tensor requires gradients
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                
                # Handle dynamic input size
                if x.size(1) != self.atom_fea_len and not self._input_adjusted:
                    # Create new embedding layer for actual input size
                    self.embedding = nn.Linear(x.size(1), self.atom_fea_len).to(x.device)
                    self._input_adjusted = True
                
                # FIXED: Ensure tensor is properly connected to computation graph
                x = x.float()
                x = self.embedding(x)
                
                # Check for valid batch size before batch norm
                if x.size(0) > 1:
                    x = self.batch_norm_input(x)
                
                for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                    x = conv(x, edge_index)
                    
                    # Safe batch norm
                    if x.size(0) > 1:
                        x = bn(x)
                    
                    x = F.relu(x)
                    x = torch.clamp(x, min=-10, max=10)
                    
                    if i < len(self.convs) - 1:
                        x = F.dropout(x, p=0.1, training=self.training)
                
                x = global_mean_pool(x, batch)
                out = self.fc(x)
                out = torch.clamp(out, min=-50, max=50)
                
                return out
        
        return CGCNNModel(**model_params)
    
    def create_megnet_model(self, model_params):
        """Create MEGNet model with improved architecture"""
        class MEGNetModel(nn.Module):
            def __init__(self, node_dim=64, edge_dim=32, global_dim=32, hidden_dim=128, n_blocks=3):
                super().__init__()
                self.node_dim = node_dim
                self.edge_dim = edge_dim
                self.global_dim = global_dim
                self.hidden_dim = hidden_dim
                self.n_blocks = n_blocks
                
                self.node_embedding = nn.Linear(node_dim, self.node_dim)
                self.edge_embedding = nn.Linear(5, self.edge_dim)
                self.global_embedding = nn.Linear(1, self.global_dim)
                
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.node_dim, self.node_dim),
                        nn.LayerNorm(self.node_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ) for _ in range(self.n_blocks)
                ])
                
                self.final_layers = nn.Sequential(
                    nn.Linear(self.node_dim + self.edge_dim + self.global_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dim, track_running_stats=False),
                    nn.Dropout(0.3),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.hidden_dim // 2, 1)
                )
                
                self._node_adjusted = False
            
            def forward(self, data):
                x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
                
                # Handle dynamic input size for nodes
                if x.size(1) != self.node_dim and not self._node_adjusted:
                    self.node_embedding = nn.Linear(x.size(1), self.node_dim).to(x.device)
                    self._node_adjusted = True
                
                x = self.node_embedding(x)
                edge_attr = self.edge_embedding(edge_attr)
                
                global_attr = torch.ones(batch.max().item() + 1, 1, device=x.device)
                global_attr = self.global_embedding(global_attr)
                
                # Residual connections in blocks
                for block in self.blocks:
                    residual = x
                    x = block(x)
                    x = x + residual
                
                node_pool = global_mean_pool(x, batch)
                
                # Proper edge pooling with correct batching
                edge_batch = batch[edge_index[0]]
                edge_pool = global_mean_pool(edge_attr, edge_batch)
                
                # Ensure dimensions match for concatenation
                batch_size = batch.max().item() + 1
                if edge_pool.size(0) != batch_size:
                    edge_pool = torch.zeros(batch_size, self.edge_dim, device=x.device)
                
                combined = torch.cat([node_pool, edge_pool, global_attr], dim=1)
                out = self.final_layers(combined)
                
                return out
        
        return MEGNetModel(**model_params)
    
    def _wrap_model_for_parallel(self, model):
        """Wrap model for distributed or data parallel training"""
        if self.is_distributed:
            # Use DistributedDataParallel for true distributed training
            model = DDP(model, device_ids=[self.device.index] if self.device.type == 'cuda' else None)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Use DataParallel for multi-GPU on single machine
            model = nn.DataParallel(model)
            
        return model
    
    def train_models(self, graph_data_list, test_size=0.2, val_size=0.2):
        """Train all models with distributed/parallel support"""
        if not PYTORCH_AVAILABLE or not graph_data_list:
            print("❌ PyTorch Geometric not available or no graph data")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING GRAPH NEURAL NETWORK MODELS")
        if self.is_distributed:
            print(f"🚀 DISTRIBUTED MODE: Using {self.world_size} GPUs")
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"📊 DATAPARALLEL MODE: Using {torch.cuda.device_count()} GPUs")
        else:
            print(f"🔧 SINGLE DEVICE MODE: {self.device}")
        print("="*60)
        
        # Split data
        n_total = len(graph_data_list)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        indices = torch.randperm(n_total)
        train_data = [graph_data_list[i] for i in indices[:n_train]]
        val_data = [graph_data_list[i] for i in indices[n_train:n_train+n_val]]
        test_data = [graph_data_list[i] for i in indices[n_train+n_val:]]
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        results = {}
        
        for name, config in self.model_configs.items():
            print(f"\n🌐 Training {name.upper()}...")
            
            model_params = config['model_params']
            training_params = config['training_params']
            
            try:
                if config['architecture'] == 'schnet_custom':
                    model = self.create_schnet_custom_model(model_params)
                elif config['architecture'] == 'cgcnn':
                    model = self.create_cgcnn_model(model_params)
                elif config['architecture'] == 'megnet_inspired':
                    model = self.create_megnet_model(model_params)
                else:
                    continue
            except Exception as e:
                print(f"❌ Error creating {name} model: {e}")
                continue
            
            model = model.to(self.device)
            
            # Wrap model for parallel training
            model = self._wrap_model_for_parallel(model)
            
            # Create data loaders with distributed sampler if needed
            if self.is_distributed:
                train_sampler = DistributedSampler(train_data, num_replicas=self.world_size, rank=self.rank)
                val_sampler = DistributedSampler(val_data, num_replicas=self.world_size, rank=self.rank, shuffle=False)
                test_sampler = DistributedSampler(test_data, num_replicas=self.world_size, rank=self.rank, shuffle=False)
                
                train_loader = DataLoader(train_data, batch_size=training_params['batch_size'], sampler=train_sampler)
                val_loader = DataLoader(val_data, batch_size=training_params['batch_size'], sampler=val_sampler)
                test_loader = DataLoader(test_data, batch_size=training_params['batch_size'], sampler=test_sampler)
            else:
                train_loader = DataLoader(train_data, batch_size=training_params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_data, batch_size=training_params['batch_size'], shuffle=False)
                test_loader = DataLoader(test_data, batch_size=training_params['batch_size'], shuffle=False)
            
            # Optimizer and scheduler
            if training_params.get('optimizer', 'adam') == 'adamw':
                optimizer = AdamW(model.parameters(), 
                                lr=training_params['learning_rate'],
                                weight_decay=training_params.get('weight_decay', 1e-4))
            else:
                optimizer = Adam(model.parameters(), 
                               lr=training_params['learning_rate'],
                               weight_decay=training_params.get('weight_decay', 1e-4))
            
            if training_params.get('scheduler', 'reduce_on_plateau') == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=training_params['epochs'])
            else:
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            start_time = time.time()
            
            # Mixed precision training for speed
            if torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
                use_amp = True
            else:
                scaler = None
                use_amp = False
            
            # Create progress bar for epochs
            if TQDM_AVAILABLE and (not self.is_distributed or self.rank == 0):
                epoch_pbar = tqdm(range(training_params['epochs']), 
                                desc=f"{name.upper()}", 
                                unit="epoch", 
                                leave=True)
            else:
                epoch_pbar = range(training_params['epochs'])
            
            for epoch in epoch_pbar:
                # Set epoch for distributed sampler
                if self.is_distributed:
                    train_loader.sampler.set_epoch(epoch)
                
                # Training
                model.train()
                total_train_loss = 0
                
                for batch in train_loader:
                    try:
                        batch = batch.to(self.device)
                        optimizer.zero_grad()
                        
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                if config['architecture'] == 'schnet_custom':
                                    # FIXED: Pass node features directly
                                    out = model(batch.x, batch.pos, batch.batch).view(-1)
                                else:
                                    out = model(batch).view(-1)
                                
                                loss = F.mse_loss(out, batch.y)
                            
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                        else:
                            if config['architecture'] == 'schnet_custom':
                                # FIXED: Pass node features directly  
                                out = model(batch.x, batch.pos, batch.batch).view(-1)
                            else:
                                out = model(batch).view(-1)
                            
                            loss = F.mse_loss(out, batch.y)
                            
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                        
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            total_train_loss += loss.item()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"  CUDA OOM, clearing cache...")
                            self._clear_cuda_cache()
                            continue
                        else:
                            print(f"  Training error: {e}")
                            continue
                
                # Validation
                model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            batch = batch.to(self.device)
                            
                            if config['architecture'] == 'schnet_custom':
                                # FIXED: Pass node features directly
                                out = model(batch.x, batch.pos, batch.batch).view(-1)
                            else:
                                out = model(batch).view(-1)
                            
                            loss = F.mse_loss(out, batch.y)
                            total_val_loss += loss.item()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                self._clear_cuda_cache()
                                continue
                            else:
                                continue
                
                avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Scheduler step based on type
                if training_params.get('scheduler', 'reduce_on_plateau') == 'cosine':
                    scheduler.step()
                else:
                    scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= 25:
                    if TQDM_AVAILABLE and (not self.is_distributed or self.rank == 0):
                        if hasattr(epoch_pbar, 'set_postfix'):
                            epoch_pbar.set_postfix({
                                'Val Loss': f'{avg_val_loss:.4f}',
                                'Best Loss': f'{best_val_loss:.4f}',
                                'Status': 'Early Stop'
                            })
                        epoch_pbar.close()
                    print(f"  Early stopping at epoch {epoch}")
                    break
                
                # Update progress bar
                if TQDM_AVAILABLE and (not self.is_distributed or self.rank == 0):
                    elapsed = time.time() - start_time
                    if hasattr(epoch_pbar, 'set_postfix'):
                        epoch_pbar.set_postfix({
                            'Val Loss': f'{avg_val_loss:.4f}',
                            'Best Loss': f'{best_val_loss:.4f}',
                            'Time': f'{elapsed:.1f}s'
                        })
                elif epoch % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Epoch {epoch}: Loss = {avg_val_loss:.4f}, Time = {elapsed:.1f}s")
            
            # Close progress bar
            if TQDM_AVAILABLE and 'epoch_pbar' in locals() and hasattr(epoch_pbar, 'close'):
                epoch_pbar.close()
            
            # Load best model and evaluate
            if 'best_model_state' in locals():
                model.load_state_dict(best_model_state)
            
            # Final evaluation
            train_metrics = self._evaluate_model(model, train_loader, config['architecture'])
            val_metrics = self._evaluate_model(model, val_loader, config['architecture'])
            test_metrics = self._evaluate_model(model, test_loader, config['architecture'])
            
            total_time = time.time() - start_time
            
            results[name] = {
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'test_r2': test_metrics['r2'],
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'test_mae': test_metrics['mae'],
                'predictions': test_metrics['predictions'],
                'targets': test_metrics['targets'],
                'training_time': total_time,
                'device_type': 'distributed' if self.is_distributed else 'parallel' if torch.cuda.device_count() > 1 else str(self.device)
            }
            
            print(f"✅ {name}: R² = {test_metrics['r2']:.3f}, Time = {total_time:.1f}s")
            self._clear_cuda_cache()
        
        self.results = results
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Print summary
        if results:
            total_training_time = sum(r['training_time'] for r in results.values())
            print(f"\n📊 Training Complete! Total time: {total_training_time:.1f}s")
            
            if self.is_distributed:
                print(f"Distributed training across {self.world_size} GPUs")
            elif torch.cuda.device_count() > 1:
                print(f"Data parallel training across {torch.cuda.device_count()} GPUs")
        
        return results
    
    def _evaluate_model(self, model, data_loader, architecture):
        """Evaluate model on data loader"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch = batch.to(self.device)
                    
                    if architecture == 'schnet_custom':
                        # FIXED: Pass node features directly
                        out = model(batch.x, batch.pos, batch.batch).view(-1)
                    else:
                        out = model(batch).view(-1)
                    
                    all_predictions.extend(out.cpu().numpy())
                    all_targets.extend(batch.y.cpu().numpy())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self._clear_cuda_cache()
                        continue
                    else:
                        continue
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Denormalize predictions and targets if normalization was applied
        if hasattr(self, 'energy_mean') and hasattr(self, 'energy_std'):
            predictions = predictions * self.energy_std + self.energy_mean
            targets = targets * self.energy_std + self.energy_mean
        
        if len(predictions) > 0 and len(targets) > 0:
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
        else:
            r2 = rmse = mae = 0.0
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def create_visualizations(self, output_dir='./distributed_results'):
        """Create comprehensive visualizations"""
        if not self.results:
            print("❌ No results to visualize. Train models first.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self._plot_training_curves(output_dir)
        self._plot_model_comparison(output_dir)
        self._plot_predictions(output_dir)
        
        print(f"📊 Visualizations saved to {output_dir}")
    
    def _plot_training_curves(self, output_dir):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
            
            train_losses = result['train_losses']
            val_losses = result['val_losses']
            training_time = result['training_time']
            device_type = result.get('device_type', 'unknown')
            
            epochs = range(1, len(train_losses) + 1)
            
            axes[i].plot(epochs, train_losses, label='Training Loss', alpha=0.8)
            axes[i].plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].set_title(f'{name.replace("_", " ").title()}\n{device_type} - {training_time:.1f}s')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distributed_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, output_dir):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(self.results.keys())
        colors = ['blue', 'green', 'orange', 'red'][:len(models)]
        
        # R² comparison
        test_r2 = [self.results[m]['test_r2'] for m in models]
        training_times = [self.results[m]['training_time'] for m in models]
        device_types = [self.results[m].get('device_type', 'unknown') for m in models]
        
        x = np.arange(len(models))
        
        bars = axes[0,0].bar(x, test_r2, alpha=0.8, color=colors)
        axes[0,0].set_ylabel('Test R² Score')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[0,0].grid(True, alpha=0.3)
        
        # Training time comparison
        axes[0,1].bar(x, training_times, alpha=0.8, color=colors)
        axes[0,1].set_ylabel('Training Time (seconds)')
        axes[0,1].set_title('Training Time Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([m.replace('_', '\n') for m in models])
        axes[0,1].grid(True, alpha=0.3)
        
        # Performance vs Time scatter
        axes[1,0].scatter(training_times, test_r2, s=100, c=colors, alpha=0.8)
        for i, model in enumerate(models):
            axes[1,0].annotate(model, (training_times[i], test_r2[i]), 
                             xytext=(5, 5), textcoords='offset points')
        
        axes[1,0].set_xlabel('Training Time (seconds)')
        axes[1,0].set_ylabel('Test R² Score')
        axes[1,0].set_title('Performance vs Training Time')
        axes[1,0].grid(True, alpha=0.3)
        
        # Device utilization summary
        device_summary = {}
        for result in self.results.values():
            device = result.get('device_type', 'unknown')
            if device not in device_summary:
                device_summary[device] = {'count': 0, 'total_time': 0}
            device_summary[device]['count'] += 1
            device_summary[device]['total_time'] += result['training_time']
        
        devices = list(device_summary.keys())
        times = [device_summary[d]['total_time'] for d in devices]
        
        axes[1,1].bar(devices, times, alpha=0.8)
        axes[1,1].set_ylabel('Total Training Time (seconds)')
        axes[1,1].set_title('Device Utilization Summary')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distributed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictions(self, output_dir):
        """Plot prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
            
            predictions = result['predictions']
            targets = result['targets']
            
            axes[i].scatter(targets, predictions, alpha=0.6, s=50, color=colors[i])
            
            # Perfect prediction line
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Metrics annotation
            r2 = result['test_r2']
            rmse = result['test_rmse']
            time_taken = result['training_time']
            device_type = result.get('device_type', 'unknown')
            
            axes[i].text(0.05, 0.95, 
                        f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nTime = {time_taken:.1f}s\n{device_type}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
            
            axes[i].set_xlabel('Actual Energy (eV)')
            axes[i].set_ylabel('Predicted Energy (eV)')
            axes[i].set_title(f'{name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distributed_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self, output_dir='./distributed_results'):
        """Save trained models and results"""
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch not available, cannot save models")
            return pd.DataFrame()
        
        if not self.results:
            print("❌ No results to save. Train models first.")
            return pd.DataFrame()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, result in self.results.items():
            model_path = output_dir / f'{name}_model.pt'
            # Handle DataParallel/DistributedDataParallel models
            if hasattr(result['model'], 'module'):
                torch.save(result['model'].module.state_dict(), model_path)
            else:
                torch.save(result['model'].state_dict(), model_path)
        
        # Save results summary
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'model': name,
                'train_r2': result['train_r2'],
                'val_r2': result['val_r2'],
                'test_r2': result['test_r2'],
                'train_rmse': result['train_rmse'],
                'val_rmse': result['val_rmse'],
                'test_rmse': result['test_rmse'],
                'train_mae': result['train_mae'],
                'val_mae': result['val_mae'],
                'test_mae': result['test_mae'],
                'training_time': result['training_time'],
                'device_type': result.get('device_type', 'unknown')
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'distributed_model_summary.csv', index=False)
        
        print(f"💾 Models and results saved to {output_dir}")
        
        return summary_df
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'is_distributed') and self.is_distributed:
            self._cleanup_distributed()


def main():
    """Main execution function with distributed support"""
    print("Distributed Graph Neural Network Models for Au Cluster Analysis")
    print("="*70)
    
    # Check PyTorch and device availability
    if PYTORCH_AVAILABLE:
        if MULTI_GPU_AVAILABLE:
            print(f"🚀 Multi-GPU training ready with {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        elif torch.cuda.is_available():
            print(f"Single GPU training ready: {torch.cuda.get_device_name(0)}")
        else:
            print("CPU training mode")
            
        if not ASE_AVAILABLE:
            print("ASE not available - needed for molecular structure handling")
    else:
        print("PyTorch Geometric not available. Please install torch-geometric")
        return None, None
    
    # Initialize analyzer with distributed support
    analyzer = DistributedGraphNeuralNetworkAnalyzer(random_state=42, use_distributed=True)
    
    # Load data
    try:
        csv_paths = [
            "./au_cluster_analysis_results/raw_coordinates.csv",
            "./au_cluster_analysis_results/descriptors.csv",
            "./raw_coordinates.csv",
            "./results/raw_coordinates.csv"
        ]
        
        csv_file = None
        for path in csv_paths:
            if Path(path).exists():
                try:
                    test_df = pd.read_csv(path, nrows=5)
                    if all(col in test_df.columns for col in ['x', 'y', 'z', 'filename']):
                        csv_file = path
                        print(f"Found CSV with coordinates: {csv_file}")
                        break
                except:
                    continue
        
        if csv_file:
            print(f"Loading data from CSV file: {csv_file}")
            analyzer.load_data(data_path=csv_file)
        else:
            print("No coordinate CSV found. Please ensure you have coordinate data available.")
            return None, None
        
        if not PYTORCH_AVAILABLE or not ASE_AVAILABLE:
            print("Cannot proceed without PyTorch Geometric and ASE")
            return analyzer, {}
        
        # Convert to graph format
        graph_data_list = analyzer.create_graph_data()
        
        if not graph_data_list:
            print("Failed to create graph data")
            return analyzer, {}
        
        # Train models (automatically uses distributed/parallel training if available)
        start_time = time.time()
        results = analyzer.train_models(graph_data_list)
        total_time = time.time() - start_time
        
        if not results:
            print("No models were successfully trained")
            return analyzer, {}
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        summary_df = analyzer.save_models()
        
        print(f"\n🎉 Distributed graph neural network analysis complete!")
        print(f"Total execution time: {total_time:.1f} seconds")
        
        if len(summary_df) > 0:
            print("\nBest performing model:")
            best_model = summary_df.loc[summary_df['test_r2'].idxmax()]
            print(f"  {best_model['model'].upper()}: R² = {best_model['test_r2']:.3f}")
            print(f"  Training time: {best_model['training_time']:.1f}s on {best_model['device_type']}")
        
        if analyzer.is_distributed:
            print(f"\nDistributed Training Performance:")
            print(f"  Training across {analyzer.world_size} GPUs")
            print(f"  Total training time: {total_time:.1f}s")
        elif torch.cuda.device_count() > 1:
            print(f"\nData Parallel Training Performance:")
            print(f"  Training across {torch.cuda.device_count()} GPUs")
            print(f"  Total training time: {total_time:.1f}s")
        
        return analyzer, results
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # Cleanup distributed training
        if hasattr(analyzer, 'is_distributed') and analyzer.is_distributed:
            analyzer._cleanup_distributed()


if __name__ == "__main__":
    analyzer, results = main()