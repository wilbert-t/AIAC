#!/usr/bin/env python3
"""
Perturbation Analysis for Au₂₀ Cluster Energy Prediction Models
Tests model robustness by applying controlled atomic displacements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import joblib
import json
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse
import sys
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try importing ASE for advanced features
try:
    from ase.atoms import Atoms
    from ase.neighborlist import NeighborList, natural_cutoffs
    HAS_ASE = True
except ImportError:
    print("⚠️ ASE not available - using basic feature calculations")
    HAS_ASE = False

# Try importing DScribe for SOAP features
try:
    from dscribe.descriptors import SOAP
    HAS_SOAP = True
except ImportError:
    print("⚠️ DScribe not available - SOAP features disabled")
    HAS_SOAP = False


class StructurePerturbation:
    """Handles atomic perturbations for Au₂₀ clusters"""
    
    def __init__(self, base_structure: np.ndarray, structure_id: str = "unknown"):
        """
        Initialize with base structure
        
        Args:
            base_structure: (20, 3) array of atomic coordinates
            structure_id: Identifier for the structure
        """
        self.base_structure = np.array(base_structure)
        self.structure_id = structure_id
        self.n_atoms = len(base_structure)
        
        # Calculate reference metrics
        self.avg_bond_length = self._calculate_avg_bond_length()
        self.min_bond_length = self._calculate_min_bond_length()
        
        print(f"📊 Structure loaded: {structure_id}")
        print(f"   Average bond length: {self.avg_bond_length:.3f} Å")
        print(f"   Minimum bond length: {self.min_bond_length:.3f} Å")
    
    def _calculate_avg_bond_length(self) -> float:
        """Calculate average Au-Au bond length"""
        distances = cdist(self.base_structure, self.base_structure)
        
        # Get upper triangle (avoid self and duplicate distances)
        upper_triangle = np.triu(distances, k=1)
        
        # Filter bonds (2.3 - 3.2 Å for Au-Au)
        bonds = upper_triangle[(upper_triangle > 2.3) & (upper_triangle < 3.2)]
        
        if len(bonds) > 0:
            return np.mean(bonds)
        else:
            # Fallback to overall average if no bonds in range
            non_zero = upper_triangle[upper_triangle > 0]
            return np.mean(non_zero) if len(non_zero) > 0 else 2.88
    
    def _calculate_min_bond_length(self) -> float:
        """Calculate minimum bond length"""
        distances = cdist(self.base_structure, self.base_structure)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        return np.min(distances)
    
    def perturb_structure(self, 
                          n_atoms_to_perturb: int, 
                          perturbation_strength: int,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply perturbation to structure
        
        Args:
            n_atoms_to_perturb: Number of atoms to perturb (1-3)
            perturbation_strength: Strength scale (1-10)
            seed: Random seed for reproducibility
            
        Returns:
            Perturbed structure and perturbation info
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Validate inputs
        n_atoms_to_perturb = min(max(1, n_atoms_to_perturb), 3)
        perturbation_strength = min(max(1, perturbation_strength), 10)
        
        # Calculate perturbation magnitude
        # Scale: 1 = 0.01Å, 10 = 0.30Å (relative to avg bond length)
        base_magnitude = 0.01 + (perturbation_strength - 1) * 0.033
        magnitude = base_magnitude * self.avg_bond_length
        
        # Select atoms to perturb
        atoms_to_perturb = np.random.choice(self.n_atoms, n_atoms_to_perturb, replace=False)
        
        # Create perturbed structure
        perturbed = self.base_structure.copy()
        perturbation_vectors = []
        
        for atom_idx in atoms_to_perturb:
            # Generate random 3D displacement
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)  # Normalize
            displacement = direction * magnitude
            
            # Apply perturbation
            perturbed[atom_idx] += displacement
            perturbation_vectors.append(displacement)
        
        # Check physical validity
        is_valid, validity_info = self._check_validity(perturbed)
        
        # Info dictionary
        info = {
            'n_atoms_perturbed': n_atoms_to_perturb,
            'perturbation_strength': perturbation_strength,
            'magnitude_angstrom': magnitude,
            'perturbed_atoms': atoms_to_perturb.tolist(),
            'displacements': np.array(perturbation_vectors),
            'max_displacement': np.max([np.linalg.norm(v) for v in perturbation_vectors]),
            'is_valid': is_valid,
            'validity_info': validity_info,
            'seed': seed
        }
        
        return perturbed, info
    
    def _check_validity(self, structure: np.ndarray) -> Tuple[bool, Dict]:
        """Check if perturbed structure is physically valid"""
        distances = cdist(structure, structure)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        
        # Validity criteria
        too_close = min_dist < 1.8  # Atoms too close
        too_far = np.any(np.min(distances, axis=1) > 5.0)  # Isolated atoms
        
        validity_info = {
            'min_distance': min_dist,
            'too_close': too_close,
            'isolated_atoms': too_far
        }
        
        is_valid = not (too_close or too_far)
        
        return is_valid, validity_info


class FeatureCalculator:
    """Calculate features for perturbed structures"""
    
    def __init__(self, feature_names: List[str], soap_params: Optional[Dict] = None):
        """
        Initialize feature calculator
        
        Args:
            feature_names: List of feature names from training
            soap_params: SOAP parameters if used in training
        """
        self.feature_names = feature_names
        self.soap_params = soap_params
        self.has_soap = HAS_SOAP and soap_params is not None
        
        if self.has_soap:
            self._init_soap()
    
    def _init_soap(self):
        """Initialize SOAP descriptor for XGBoost feature compatibility"""
        # Use parameters that match the training setup
        default_params = {
            'r_cut': 4.48,
            'n_max': 8, 
            'l_max': 8,
            'sigma': 0.5
        }
        
        if self.soap_params:
            soap_params = {**default_params, **self.soap_params}
        else:
            soap_params = default_params
            
        self.soap = SOAP(
            species=['Au'],
            r_cut=soap_params['r_cut'],
            n_max=soap_params['n_max'],
            l_max=soap_params['l_max'],
            sigma=soap_params['sigma'],
            periodic=False,
            sparse=False,
            average='inner'
        )
        print(f"✅ SOAP initialized with r_cut={soap_params['r_cut']}, n_max={soap_params['n_max']}, l_max={soap_params['l_max']}")
    
    def calculate_features(self, structure: np.ndarray) -> np.ndarray:
        """Calculate features matching XGBoost training format (30 features)"""
        features = {}
        
        # Basic geometric features (15 features)
        features.update(self._calculate_basic_features(structure))
        
        # SOAP features if available (15 SOAP PCs)
        if self.has_soap and any('soap_pc' in name for name in self.feature_names):
            soap_features = self._calculate_soap_features(structure)
            features.update(soap_features)
        
        # Match feature order from training (critical for XGBoost)
        feature_vector = []
        for name in self.feature_names:
            if name in features:
                feature_vector.append(features[name])
            else:
                # Use default value for missing features
                if 'soap' in name:
                    feature_vector.append(0.0)  # SOAP default
                elif 'bond' in name:
                    feature_vector.append(2.88)  # Au-Au bond default
                elif 'coordination' in name:
                    feature_vector.append(8.0)  # Coordination default
                else:
                    feature_vector.append(0.0)  # General default
        
        if len(feature_vector) != len(self.feature_names):
            print(f"⚠️ Feature mismatch: expected {len(self.feature_names)}, got {len(feature_vector)}")
        
        return np.array(feature_vector)
    
    def _calculate_basic_features(self, structure: np.ndarray) -> Dict:
        """Calculate basic structural features"""
        features = {}
        
        # Calculate distances
        distances = cdist(structure, structure)
        np.fill_diagonal(distances, np.inf)
        
        # Bond statistics (2.3-3.2 Å)
        bonds = []
        for i in range(len(structure)):
            for j in range(i+1, len(structure)):
                dist = np.linalg.norm(structure[i] - structure[j])
                if 2.3 <= dist <= 3.2:
                    bonds.append(dist)
        
        if bonds:
            features['mean_bond_length'] = np.mean(bonds)
            features['std_bond_length'] = np.std(bonds)
            features['n_bonds'] = len(bonds)
        else:
            features['mean_bond_length'] = 2.88
            features['std_bond_length'] = 0.1
            features['n_bonds'] = 0
        
        # Coordination numbers
        coordination = []
        for i in range(len(structure)):
            neighbors = np.sum((distances[i] > 2.3) & (distances[i] < 3.2))
            coordination.append(neighbors)
        
        features['mean_coordination'] = np.mean(coordination)
        features['std_coordination'] = np.std(coordination)
        features['max_coordination'] = np.max(coordination)
        
        # Geometric properties
        center = np.mean(structure, axis=0)
        centered = structure - center
        
        # Radius of gyration
        rg = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        features['radius_of_gyration'] = rg
        
        # Asphericity
        gyration_tensor = np.dot(centered.T, centered) / len(structure)
        eigenvals = np.sort(np.linalg.eigvals(gyration_tensor))[::-1]
        
        if np.sum(eigenvals) > 0:
            asphericity = eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])
            features['asphericity'] = asphericity / np.sum(eigenvals)
        else:
            features['asphericity'] = 0.0
        
        # Surface fraction (approximate)
        features['surface_fraction'] = np.sum(np.array(coordination) < 8) / len(structure)
        
        # Bounding box
        ranges = np.max(structure, axis=0) - np.min(structure, axis=0)
        features['x_range'] = ranges[0]
        features['y_range'] = ranges[1]
        features['z_range'] = ranges[2]
        features['anisotropy'] = np.max(ranges) / (np.min(ranges) + 1e-8)
        
        # Volume and compactness
        volume = np.prod(ranges)
        features['compactness'] = len(structure) / (volume + 1e-8)
        features['bond_variance'] = np.var(bonds) if bonds else 0.0
        
        return features
    
    def _calculate_soap_features(self, structure: np.ndarray) -> Dict:
        """Calculate SOAP descriptors and return principal components"""
        features = {}
        
        if not HAS_SOAP:
            # Return default SOAP PCs if SOAP not available
            for i in range(1, 16):  # soap_pc_1 to soap_pc_15
                features[f'soap_pc_{i}'] = 0.0
            return features
        
        try:
            atoms = Atoms('Au' * len(structure), positions=structure)
            soap_matrix = self.soap.create(atoms)
            
            # Ensure soap_matrix is 2D
            if soap_matrix.ndim == 1:
                soap_matrix = soap_matrix.reshape(1, -1)
            
            # Calculate average SOAP vector
            soap_avg = np.mean(soap_matrix, axis=0)
            
            # Ensure soap_avg is 1D array
            if isinstance(soap_avg, np.ndarray):
                soap_features = soap_avg
            else:
                soap_features = np.array([soap_avg])
            
            # For XGBoost compatibility, use first 15 components as PCs
            # (This assumes the original training used PCA on SOAP descriptors)
            n_features = min(15, len(soap_features))
            for i in range(n_features):
                features[f'soap_pc_{i+1}'] = float(soap_features[i])
            
            # Fill remaining PCs if needed
            for i in range(n_features, 15):
                features[f'soap_pc_{i+1}'] = 0.0
                
            print(f"✅ Calculated SOAP features: {len([k for k in features.keys() if 'soap' in k])} components")
                
        except Exception as e:
            print(f"⚠️ SOAP calculation failed: {e}")
            # Return default SOAP PCs
            for i in range(1, 16):
                features[f'soap_pc_{i}'] = 0.0
        
        return features


class ModelEvaluator:
    """Evaluate models on perturbed structures"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.predictions = []
    
    def load_models(self, model_paths: Dict[str, str]):
        """
        Load trained models with metadata
        
        Args:
            model_paths: Dictionary of {model_type: path_to_model}
        """
        for model_type, path in model_paths.items():
            try:
                # Load model
                model = joblib.load(path)
                self.models[model_type] = model
                
                # XGBoost and tree models don't need scaling
                if any(keyword in model_type.lower() for keyword in ['xgb', 'random_forest', 'lightgbm', 'catboost', 'gradient_boost', 'tree']):
                    self.scalers[model_type] = None
                else:
                    # Try to load scaler for other models
                    scaler_path = Path(path).parent / f"{Path(path).stem.replace('_model', '')}_scaler.joblib"
                    if scaler_path.exists():
                        self.scalers[model_type] = joblib.load(scaler_path)
                    else:
                        print(f"⚠️ No scaler found for {model_type}, disabling scaling")
                        self.scalers[model_type] = None
                
                # Load metadata (especially important for XGBoost)
                metadata_path = Path(path).parent / f"{Path(path).stem.replace('_model', '')}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        training_info = metadata.get('training_info', {})
                        self.feature_names[model_type] = training_info.get('feature_names', [])
                        
                        print(f"✅ Loaded {model_type} model with {len(self.feature_names[model_type])} features")
                        if model_type == 'xgboost':
                            print(f"   Performance: R² = {metadata.get('performance', {}).get('test_r2', 'N/A'):.3f}")
                            print(f"   Features include SOAP descriptors: {'soap_pc_1' in self.feature_names[model_type]}")
                else:
                    print(f"⚠️ No metadata found for {model_type}, using default features")
                    self.feature_names[model_type] = []
                
                print(f"✅ Loaded {model_type} model from {path}")
                
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")
                import traceback
                traceback.print_exc()
    
    def predict_energy(self, structure: np.ndarray, model_type: str, 
                      feature_calculator: FeatureCalculator) -> float:
        """Predict energy for a structure"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        # Calculate features
        features = feature_calculator.calculate_features(structure)
        
        # Scale features only if scaler exists (not for tree models)
        if self.scalers.get(model_type) is not None:
            features = self.scalers[model_type].transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # Predict
        energy = self.models[model_type].predict(features)[0]
        
        return energy
    
    def evaluate_perturbations(self, 
                              base_structure: np.ndarray,
                              perturbed_structures: List[Tuple[np.ndarray, Dict]],
                              reference_energy: float) -> pd.DataFrame:
        """
        Evaluate all models on perturbed structures
        
        Returns:
            DataFrame with results
        """
        results = []
        
        # Create feature calculator for the model
        if not self.models:
            raise ValueError("No models loaded for evaluation")
        
        model_type = list(self.models.keys())[0]  # Use the primary model (XGBoost)
        feature_names = self.feature_names.get(model_type, [])
        
        if not feature_names:
            # Use default XGBoost feature names
            feature_names = [
                'mean_bond_length', 'std_bond_length', 'n_bonds', 'mean_coordination', 
                'std_coordination', 'max_coordination', 'radius_of_gyration', 'asphericity',
                'surface_fraction', 'x_range', 'y_range', 'z_range', 'anisotropy', 
                'compactness', 'bond_variance'
            ] + [f'soap_pc_{i}' for i in range(1, 16)]
            print(f"⚠️ Using default XGBoost feature names ({len(feature_names)} features)")
        
        feature_calculator = FeatureCalculator(feature_names)
        print(f"✅ Using {model_type} model with {len(feature_names)} features")
        
        # Predict base energy
        try:
            base_energy = self.predict_energy(base_structure, model_type, feature_calculator)
            print(f"📊 Base structure energy prediction: {base_energy:.6f} eV")
        except Exception as e:
            print(f"❌ Failed to predict base energy: {e}")
            base_energy = np.nan
        
        # Evaluate perturbations
        for i, (perturbed_structure, info) in enumerate(perturbed_structures):
            result = {
                'perturbation_id': i,
                'n_atoms_perturbed': info['n_atoms_perturbed'],
                'perturbation_strength': info['perturbation_strength'],
                'magnitude_angstrom': info['magnitude_angstrom'],
                'is_valid': info['is_valid'],
                'reference_energy': reference_energy
            }
            
            # Predict with the model
            try:
                perturbed_energy = self.predict_energy(
                    perturbed_structure, model_type, feature_calculator
                )
                
                result[f'{model_type}_base_energy'] = base_energy
                result[f'{model_type}_perturbed_energy'] = perturbed_energy
                result[f'{model_type}_delta_energy'] = perturbed_energy - base_energy
                result[f'{model_type}_sensitivity'] = abs(perturbed_energy - base_energy) / info['magnitude_angstrom']
                
            except Exception as e:
                print(f"⚠️ Failed to predict for {model_type} on perturbation {i}: {e}")
                result[f'{model_type}_perturbed_energy'] = np.nan
                result[f'{model_type}_delta_energy'] = np.nan
                result[f'{model_type}_sensitivity'] = np.nan
            
            results.append(result)
        
        return pd.DataFrame(results)


class PerturbationAnalyzer:
    """Main analyzer for perturbation study"""
    
    def __init__(self, output_dir: str = './perturbation_results'):
        """Initialize analyzer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.structure_perturbator = None
        self.evaluator = ModelEvaluator()
        self.results = None
    
    def load_reference_structure(self, structure_file: str, structure_id: str = None):
        """Load reference structure (lowest energy)"""
        # Try different file formats
        if structure_file.endswith('.xyz'):
            structure, energy = self._load_xyz_file(structure_file)
            structure_id = Path(structure_file).stem
        elif structure_file.endswith('.csv'):
            structure, energy, structure_id = self._load_from_csv(structure_file, structure_id)
        else:
            raise ValueError(f"Unsupported file format: {structure_file}")
        
        self.structure_perturbator = StructurePerturbation(structure, structure_id)
        self.reference_energy = energy
        
        print(f"✅ Loaded reference structure: {structure_id}")
        print(f"   Reference energy: {energy:.6f} eV")
    
    def _load_xyz_file(self, filepath: str) -> Tuple[np.ndarray, float]:
        """Load structure from XYZ file"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        energy = float(lines[1].strip())
        
        coordinates = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append([x, y, z])
         
        return np.array(coordinates), energy
    
    def _load_from_csv(self, filepath: str, structure_id: str = None) -> Tuple[np.ndarray, float, str]:
        """Load structure from CSV dataset"""
        df = pd.read_csv(filepath)
        
        # Check if this is descriptors.csv (Task 2 format)
        if 'filename' in df.columns and 'energy' in df.columns:
            # This is descriptors.csv format
            if structure_id:
                if not structure_id.endswith('.xyz'):
                    structure_id += '.xyz'
                # Find the structure in descriptors
                row = df[df['filename'] == structure_id]
                if len(row) == 0:
                    raise ValueError(f"Structure {structure_id} not found in descriptors.csv")
                row = row.iloc[0]
                energy = row['energy']
                
                # Load coordinates from original XYZ file
                try:
                    coordinates = load_coordinates_from_xyz(structure_id)
                    print(f"✅ Loaded coordinates from XYZ file: {structure_id}")
                    return coordinates, energy, structure_id
                except FileNotFoundError:
                    raise ValueError(f"Could not load coordinates for {structure_id}")
            else:
                # Use the first (best) structure
                row = df.iloc[0]
                structure_id = row['filename']
                energy = row['energy']
                
                try:
                    coordinates = load_coordinates_from_xyz(structure_id)
                    print(f"✅ Loaded coordinates from XYZ file: {structure_id}")
                    return coordinates, energy, structure_id
                except FileNotFoundError:
                    raise ValueError(f"Could not load coordinates for {structure_id}")
        
        else:
            # Original format with coordinate columns
            # Find the structure (default to first row - lowest energy)
            if 'structure_id' in df.columns:
                if structure_id and structure_id in df['structure_id'].values:
                    row = df[df['structure_id'] == structure_id].iloc[0]
                else:
                    # Use first row (should be lowest energy from previous analysis)
                    row = df.iloc[0]
                    structure_id = row.get('structure_id', 'unknown')
            else:
                row = df.iloc[0]
                structure_id = 'unknown'
            
            # Extract coordinates
            coordinates = []
            for i in range(1, 21):  # Au20
                if f'atom_{i}_x' in row:
                    x = row[f'atom_{i}_x']
                    y = row[f'atom_{i}_y']
                    z = row[f'atom_{i}_z']
                    coordinates.append([x, y, z])
            
            # Try different energy column names
            energy = 0.0
            for energy_col in ['predicted_energy', 'actual_energy', 'energy']:
                if energy_col in row:
                    energy = row[energy_col]
                    break
            
            return np.array(coordinates), energy, structure_id
    
    def run_perturbation_analysis(self,
                                 n_atoms_to_perturb: int,
                                 perturbation_strength: int,
                                 n_trials: int = 10):
        """
        Run comprehensive perturbation analysis
        
        Args:
            n_atoms_to_perturb: Number of atoms to perturb (1-3)
            perturbation_strength: Strength scale (1-10)
            n_trials: Number of perturbation trials
        """
        print(f"\n{'='*60}")
        print(f"PERTURBATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Atoms to perturb: {n_atoms_to_perturb}")
        print(f"  Perturbation strength: {perturbation_strength}/10")
        print(f"  Number of trials: {n_trials}")
        
        # Generate perturbations
        perturbed_structures = []
        for trial in range(n_trials):
            perturbed, info = self.structure_perturbator.perturb_structure(
                n_atoms_to_perturb, perturbation_strength, seed=trial
            )
            perturbed_structures.append((perturbed, info))
        
        # Evaluate with all models
        self.results = self.evaluator.evaluate_perturbations(
            self.structure_perturbator.base_structure,
            perturbed_structures,
            self.reference_energy
        )
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Save results
        self._save_results(n_atoms_to_perturb, perturbation_strength)
        
        # Create visualizations
        self._create_visualizations(n_atoms_to_perturb, perturbation_strength)
    
    def _calculate_statistics(self):
        """Calculate error metrics and statistics"""
        print(f"\n📊 RESULTS SUMMARY")
        print(f"{'='*60}")
        
        model_types = [col.replace('_perturbed_energy', '') 
                      for col in self.results.columns 
                      if col.endswith('_perturbed_energy')]
        
        for model_type in model_types:
            # Get predictions
            perturbed = self.results[f'{model_type}_perturbed_energy'].dropna()
            base = self.results[f'{model_type}_base_energy'].dropna()
            delta = self.results[f'{model_type}_delta_energy'].dropna()
            sensitivity = self.results[f'{model_type}_sensitivity'].dropna()
            
            if len(perturbed) > 0:
                # Calculate metrics
                mae_from_ref = np.mean(np.abs(perturbed - self.reference_energy))
                rmse_from_ref = np.sqrt(np.mean((perturbed - self.reference_energy)**2))
                
                print(f"\n{model_type.upper()}:")
                print(f"  Base energy prediction: {base.iloc[0]:.6f} eV")
                print(f"  Mean perturbed energy: {perturbed.mean():.6f} eV")
                print(f"  Energy change (mean): {delta.mean():.6f} eV")
                print(f"  Energy change (std): {delta.std():.6f} eV")
                print(f"  Sensitivity (mean): {sensitivity.mean():.3f} eV/Å")
                print(f"  MAE from reference: {mae_from_ref:.6f} eV")
                print(f"  RMSE from reference: {rmse_from_ref:.6f} eV")
    
    def _save_results(self, n_atoms: int, strength: int):
        """Save results to files"""
        # Save detailed results
        filename = f'perturbation_results_n{n_atoms}_s{strength}.csv'
        self.results.to_csv(self.output_dir / filename, index=False)
        print(f"\n💾 Results saved to {self.output_dir / filename}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_atoms_perturbed': n_atoms,
            'perturbation_strength': strength,
            'n_trials': len(self.results),
            'reference_energy': self.reference_energy,
            'models_evaluated': list(self.evaluator.models.keys())
        }
        
        with open(self.output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _create_visualizations(self, n_atoms: int, strength: int):
        """Create visualization plots"""
        print("📊 Creating visualizations...")
        
        model_types = [col.replace('_perturbed_energy', '') 
                      for col in self.results.columns 
                      if col.endswith('_perturbed_energy')]
        
        if not model_types:
            print("⚠️ No models to visualize")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Energy distribution
        ax = axes[0, 0]
        for model_type in model_types:
            energies = self.results[f'{model_type}_perturbed_energy'].dropna()
            if len(energies) > 0:
                ax.hist(energies, alpha=0.5, label=model_type, bins=15)
        
        ax.axvline(self.reference_energy, color='red', linestyle='--', 
                  label='Reference', linewidth=2)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Distribution Across Perturbations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Energy change vs perturbation
        ax = axes[0, 1]
        for model_type in model_types:
            delta = self.results[f'{model_type}_delta_energy'].dropna()
            if len(delta) > 0:
                x = range(len(delta))
                ax.scatter(x, delta, alpha=0.6, label=model_type, s=30)
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Perturbation Trial')
        ax.set_ylabel('Energy Change (eV)')
        ax.set_title('Energy Changes from Base Structure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Model sensitivity comparison
        ax = axes[1, 0]
        sensitivities = []
        labels = []
        
        for model_type in model_types:
            sens = self.results[f'{model_type}_sensitivity'].dropna()
            if len(sens) > 0:
                sensitivities.append(sens.values)
                labels.append(model_type)
        
        if sensitivities:
            bp = ax.boxplot(sensitivities, labels=labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Sensitivity (eV/Å)')
        ax.set_title('Model Sensitivity Comparison')
        ax.grid(True, alpha=0.3)
        
        # 4. Perturbation magnitude vs energy change
        ax = axes[1, 1]
        magnitudes = self.results['magnitude_angstrom'].values
        
        for model_type in model_types:
            delta = self.results[f'{model_type}_delta_energy'].dropna()
            if len(delta) > 0:
                valid_indices = ~self.results[f'{model_type}_delta_energy'].isna()
                mags = magnitudes[valid_indices]
                ax.scatter(mags, np.abs(delta), alpha=0.6, label=model_type, s=30)
        
        ax.set_xlabel('Perturbation Magnitude (Å)')
        ax.set_ylabel('|Energy Change| (eV)')
        ax.set_title('Energy Change vs Perturbation Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'perturbation_analysis_n{n_atoms}_s{strength}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional plots for model comparison
        self._create_model_comparison_plots(n_atoms, strength)
        
        print(f"📊 Visualizations saved to {self.output_dir}")
    
    def _create_model_comparison_plots(self, n_atoms: int, strength: int):
        """Create detailed model comparison visualizations"""
        model_types = [col.replace('_perturbed_energy', '') 
                      for col in self.results.columns 
                      if col.endswith('_perturbed_energy')]
        
        if len(model_types) < 2:
            print("⚠️ Need at least 2 models for comparison plots")
            return
        
        # Model correlation matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Energy predictions correlation
        energy_data = {}
        for model_type in model_types:
            energies = self.results[f'{model_type}_perturbed_energy'].dropna()
            if len(energies) > 0:
                energy_data[model_type] = energies.values
        
        if len(energy_data) >= 2:
            # Find common indices
            min_length = min(len(v) for v in energy_data.values())
            energy_df = pd.DataFrame({k: v[:min_length] for k, v in energy_data.items()})
            
            # Correlation heatmap
            corr_matrix = energy_df.corr()
            im1 = ax1.imshow(corr_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
            ax1.set_xticks(range(len(model_types)))
            ax1.set_yticks(range(len(model_types)))
            ax1.set_xticklabels(model_types, rotation=45)
            ax1.set_yticklabels(model_types)
            ax1.set_title('Model Energy Prediction Correlation')
            
            # Add correlation values
            for i in range(len(model_types)):
                for j in range(len(model_types)):
                    ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im1, ax=ax1)
        
        # Sensitivity comparison
        sensitivity_data = []
        labels = []
        for model_type in model_types:
            sens = self.results[f'{model_type}_sensitivity'].dropna()
            if len(sens) > 0:
                sensitivity_data.append(sens.values)
                labels.append(model_type)
        
        if sensitivity_data:
            bp = ax2.violinplot(sensitivity_data, positions=range(len(labels)), 
                              showmeans=True, showmedians=True)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45)
            ax2.set_ylabel('Sensitivity (eV/Å)')
            ax2.set_title('Model Sensitivity Distribution')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'model_comparison_n{n_atoms}_s{strength}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_structure_visualization(self, structure_before: np.ndarray, 
                                     structure_after: np.ndarray, 
                                     perturbation_info: Dict,
                                     structure_id: str,
                                     output_dir: Path) -> str:
        """Create 3D before/after visualization of atomic perturbation"""
        
        fig = plt.figure(figsize=(15, 6))
        
        # Before perturbation
        ax1 = fig.add_subplot(121, projection='3d')
        x, y, z = structure_before.T
        ax1.scatter(x, y, z, c='gold', s=100, alpha=0.8, label='Au atoms')
        ax1.set_title(f'Before Perturbation\n{structure_id}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        
        # Make axes equal
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # After perturbation
        ax2 = fig.add_subplot(122, projection='3d')
        x, y, z = structure_after.T
        
        # Color-code: perturbed atoms in red, others in gold
        colors = ['red' if i in perturbation_info['perturbed_atoms'] else 'gold' 
                  for i in range(len(structure_after))]
        sizes = [150 if i in perturbation_info['perturbed_atoms'] else 100 
                 for i in range(len(structure_after))]
        
        ax2.scatter(x, y, z, c=colors, s=sizes, alpha=0.8)
        ax2.set_title(f'After Perturbation\n{perturbation_info["n_atoms_perturbed"]} atoms, '
                      f'strength {perturbation_info["perturbation_strength"]}/10\n'
                      f'Max displacement: {perturbation_info["max_displacement"]:.3f} Å', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Y (Å)')
        ax2.set_zlabel('Z (Å)')
        
        # Make axes equal
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add displacement vectors
        for atom_idx, displacement in zip(perturbation_info['perturbed_atoms'], 
                                         perturbation_info['displacements']):
            start = structure_before[atom_idx]
            end = structure_after[atom_idx]
            ax2.quiver(start[0], start[1], start[2],
                      displacement[0], displacement[1], displacement[2],
                      color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                                markersize=10, label='Unperturbed Au', alpha=0.8),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                markersize=12, label='Perturbed Au', alpha=0.8),
                          Line2D([0], [0], color='blue', linewidth=2, 
                                label='Displacement', alpha=0.8)]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        filename = f'structure_perturbation_{structure_id.replace(".xyz", "")}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def calculate_model_agreement(self) -> Dict:
        """Calculate agreement metrics between models"""
        model_types = [col.replace('_perturbed_energy', '') 
                      for col in self.results.columns 
                      if col.endswith('_perturbed_energy')]
        
        if len(model_types) < 2:
            return {}
        
        agreement_metrics = {}
        
        # Collect all predictions
        predictions = {}
        for model_type in model_types:
            pred = self.results[f'{model_type}_perturbed_energy'].dropna()
            if len(pred) > 0:
                predictions[model_type] = pred.values
        
        if len(predictions) >= 2:
            # Calculate pairwise correlations
            models = list(predictions.keys())
            correlations = {}
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models[i+1:], i+1):
                    min_len = min(len(predictions[model1]), len(predictions[model2]))
                    if min_len > 1:
                        corr = np.corrcoef(predictions[model1][:min_len], 
                                         predictions[model2][:min_len])[0, 1]
                        correlations[f'{model1}_vs_{model2}'] = corr
            
            agreement_metrics['correlations'] = correlations
            agreement_metrics['mean_correlation'] = np.mean(list(correlations.values()))
            
            # Calculate prediction variance across models
            all_preds = np.array([predictions[m] for m in models])
            min_samples = min(pred.shape[0] for pred in all_preds)
            trimmed_preds = np.array([pred[:min_samples] for pred in all_preds])
            
            agreement_metrics['prediction_variance'] = np.var(trimmed_preds, axis=0)
            agreement_metrics['mean_prediction_variance'] = np.mean(agreement_metrics['prediction_variance'])
            
            # Identify consensus vs divergent predictions
            variance_threshold = np.percentile(agreement_metrics['prediction_variance'], 75)
            agreement_metrics['high_agreement_indices'] = np.where(
                agreement_metrics['prediction_variance'] < variance_threshold
            )[0].tolist()
            agreement_metrics['low_agreement_indices'] = np.where(
                agreement_metrics['prediction_variance'] >= variance_threshold
            )[0].tolist()
        
        return agreement_metrics

    def select_representative_structures(self, all_results: pd.DataFrame) -> Dict:
        """Select one representative structure from each category"""
        
        if len(all_results) == 0:
            raise ValueError("No results available for structure selection")
        
        # Sort by energy (stability)
        sorted_by_energy = all_results.sort_values('reference_energy')
        
        # Sort by sensitivity (robustness)
        sorted_by_sensitivity = all_results.sort_values('mean_sensitivity')
        
        representatives = {
            'most_stable': {
                'id': sorted_by_energy.iloc[0]['structure_id'],
                'energy': sorted_by_energy.iloc[0]['reference_energy'],
                'sensitivity': sorted_by_energy.iloc[0]['mean_sensitivity'],
                'description': 'Lowest energy structure (thermodynamically most stable)'
            },
            'most_robust': {
                'id': sorted_by_sensitivity.iloc[0]['structure_id'],
                'energy': sorted_by_sensitivity.iloc[0]['reference_energy'], 
                'sensitivity': sorted_by_sensitivity.iloc[0]['mean_sensitivity'],
                'description': 'Lowest sensitivity to perturbations (most robust)'
            },
            'most_sensitive': {
                'id': sorted_by_sensitivity.iloc[-1]['structure_id'],
                'energy': sorted_by_sensitivity.iloc[-1]['reference_energy'],
                'sensitivity': sorted_by_sensitivity.iloc[-1]['mean_sensitivity'],
                'description': 'Highest sensitivity to perturbations (least robust)'
            }
        }
        
        # Ensure we don't have duplicates - if structures are the same, pick alternatives
        structure_ids = [rep['id'] for rep in representatives.values()]
        if len(set(structure_ids)) < len(structure_ids):
            # Handle duplicates by picking from different quartiles
            n_structures = len(sorted_by_sensitivity)
            representatives['most_robust']['id'] = sorted_by_sensitivity.iloc[0]['structure_id']
            representatives['most_sensitive']['id'] = sorted_by_sensitivity.iloc[-1]['structure_id']
            
            # For most stable, pick from middle of sensitivity range if it's the same as others
            middle_idx = n_structures // 2
            if representatives['most_stable']['id'] in [representatives['most_robust']['id'], 
                                                      representatives['most_sensitive']['id']]:
                representatives['most_stable'] = {
                    'id': sorted_by_sensitivity.iloc[middle_idx]['structure_id'],
                    'energy': sorted_by_sensitivity.iloc[middle_idx]['reference_energy'],
                    'sensitivity': sorted_by_sensitivity.iloc[middle_idx]['mean_sensitivity'],
                    'description': 'Moderate sensitivity (balanced structure)'
                }
        
        return representatives

    def create_representative_visualizations(self, representatives: Dict, 
                                           n_atoms: int = 2, 
                                           strength: int = 5) -> Dict[str, str]:
        """Create visualizations for representative structures from each category"""
        
        print(f"\n🎨 CREATING REPRESENTATIVE STRUCTURE VISUALIZATIONS")
        print("=" * 60)
        
        viz_output_dir = self.output_dir / 'representative_visualizations'
        viz_output_dir.mkdir(exist_ok=True)
        
        visualization_files = {}
        
        for category, structure_info in representatives.items():
            print(f"\n📊 Creating visualization for {category}:")
            print(f"   Structure: {structure_info['id']}")
            print(f"   Energy: {structure_info['energy']:.6f} eV")
            print(f"   Sensitivity: {structure_info['sensitivity']:.4f} eV/Å")
            print(f"   Description: {structure_info['description']}")
            
            try:
                # Load the specific structure coordinates
                structure_coords = self._load_structure_coordinates(structure_info['id'])
                
                # Create temporary perturbator for this structure
                temp_perturbator = StructurePerturbation(structure_coords, structure_info['id'])
                
                # Generate one perturbation for visualization (reproducible)
                perturbed_structure, perturbation_info = temp_perturbator.perturb_structure(
                    n_atoms_to_perturb=n_atoms,
                    perturbation_strength=strength,
                    seed=42  # Reproducible
                )
                
                # Create 3D visualization
                filename = self.create_structure_visualization(
                    structure_before=structure_coords,
                    structure_after=perturbed_structure,
                    perturbation_info=perturbation_info,
                    structure_id=structure_info['id'],
                    output_dir=viz_output_dir
                )
                
                visualization_files[category] = filename
                print(f"   ✅ Saved: {filename}")
                
            except Exception as e:
                print(f"   ❌ Failed to create visualization: {e}")
                visualization_files[category] = None
        
        # Create summary document
        self._create_visualization_summary(representatives, visualization_files, 
                                         viz_output_dir, n_atoms, strength)
        
        print(f"\n🎉 Representative visualizations complete!")
        print(f"📁 Saved to: {viz_output_dir}")
        
        return visualization_files

    def _load_structure_coordinates(self, structure_id: str) -> np.ndarray:
        """Load coordinates for a specific structure ID"""
        
        # Handle different structure ID formats
        # Convert 'structure_729' to '729.xyz' or keep '729.xyz' as is
        if structure_id.startswith('structure_'):
            # Extract number from 'structure_729' -> '729'
            number = structure_id.replace('structure_', '')
            xyz_filename = f"{number}.xyz"
        elif structure_id.endswith('.xyz'):
            xyz_filename = structure_id
        else:
            # Assume it's just a number
            xyz_filename = f"{structure_id}.xyz"
        
        # Try original XYZ files first
        xyz_file = Path(f"data/Au20_OPT_1000/{xyz_filename}")
        if xyz_file.exists():
            return self._load_xyz_coordinates(xyz_file)
        
        # Try loading from CSV datasets
        csv_files = [
            'tree_models_results/top_20_stable_structures.csv',
            'task2/dataset_elite.csv',
            'au_cluster_analysis_results/descriptors.csv'
        ]
        
        for csv_file in csv_files:
            if Path(csv_file).exists():
                try:
                    df = pd.read_csv(csv_file)
                    # Try both original structure_id and xyz_filename
                    if structure_id in df['structure_id'].values or xyz_filename in df['structure_id'].values:
                        # Try the XYZ file
                        if xyz_file.exists():
                            return self._load_xyz_coordinates(xyz_file)
                except:
                    continue
        
        raise FileNotFoundError(f"Could not find coordinates for structure {structure_id} (tried {xyz_filename})")

    def _load_xyz_coordinates(self, filepath: Path) -> np.ndarray:
        """Load coordinates from XYZ file"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        coordinates = []
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append([x, y, z])
        
        return np.array(coordinates)

    def _create_visualization_summary(self, representatives: Dict, 
                                    visualization_files: Dict[str, str],
                                    output_dir: Path, n_atoms: int, strength: int):
        """Create a summary document for the representative visualizations"""
        
        summary_file = output_dir / 'representative_structures_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("REPRESENTATIVE STRUCTURE VISUALIZATIONS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("VISUALIZATION PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Atoms Perturbed: {n_atoms}\n")
            f.write(f"Perturbation Strength: {strength}/10\n")
            f.write(f"Random Seed: 42 (reproducible)\n\n")
            
            f.write("REPRESENTATIVE STRUCTURES\n")
            f.write("-" * 40 + "\n")
            
            for category, structure_info in representatives.items():
                f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                f.write(f"  Structure ID: {structure_info['id']}\n")
                f.write(f"  Energy: {structure_info['energy']:.6f} eV\n")
                f.write(f"  Sensitivity: {structure_info['sensitivity']:.4f} eV/Å\n")
                f.write(f"  Description: {structure_info['description']}\n")
                
                if visualization_files.get(category):
                    f.write(f"  Visualization: {visualization_files[category]}\n")
                else:
                    f.write(f"  Visualization: Failed to generate\n")
            
            f.write(f"\nVISUALIZATION FEATURES\n")
            f.write("-" * 40 + "\n")
            f.write("• Gold spheres: Unperturbed Au atoms\n")
            f.write("• Red spheres: Perturbed Au atoms (larger)\n")
            f.write("• Blue arrows: Displacement vectors\n")
            f.write("• Side-by-side: Before and after comparison\n")
            f.write("• Equal axes: Consistent scale for comparison\n\n")
            
            f.write("USAGE IN DOCUMENT\n")
            f.write("-" * 40 + "\n")
            f.write("These visualizations can be used to demonstrate:\n")
            f.write("• How different structure types respond to perturbations\n")
            f.write("• The relationship between stability and robustness\n")
            f.write("• Visual evidence of model sensitivity differences\n")
            f.write("• Clear before/after structural changes\n")
        
        print(f"📄 Summary saved to: {summary_file}")


def find_available_models() -> Dict[str, str]:
    """Automatically discover XGBoost model from tree models results"""
    model_paths = {}
    
    # Prioritize XGBoost model from tree models (best performer)
    xgboost_path = Path('tree_models_results/trained_models/xgboost_model.joblib')
    if xgboost_path.exists():
        model_paths['xgboost'] = str(xgboost_path)
        print(f"✅ Found XGBoost model: {xgboost_path}")
        return model_paths
    
    # Fallback to other tree models
    tree_model_files = [
        'tree_models_results/trained_models/random_forest_model.joblib',
        'tree_models_results/trained_models/lightgbm_model.joblib',
        'tree_models_results/trained_models/catboost_model.joblib',
        'tree_models_results/trained_models/gradient_boosting_model.joblib'
    ]
    
    for model_file in tree_model_files:
        if Path(model_file).exists():
            name = Path(model_file).stem.replace('_model', '')
            model_paths[name] = model_file
            print(f"✅ Found fallback model: {name} at {model_file}")
            break  # Use only one model for focused analysis
    
    # Last resort: search other directories
    if not model_paths:
        search_dirs = [
            'kernel_models_analysis/saved_models',
            'linear_models_results/trained_models',
            './trained_models'
        ]
        
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if search_path.exists():
                for model_file in search_path.glob('*model*.joblib'):
                    name = model_file.stem.replace('_model', '')
                    model_paths[name] = str(model_file)
                    print(f"⚠️ Using fallback: {name} from {model_file}")
                    break
                if model_paths:
                    break
    
    return model_paths


def load_reference_structure_auto() -> Tuple[str, str]:
    """Automatically find the best reference structure from tree models results"""
    # Prioritize tree models results (best performing XGBoost model)
    priority_files = [
        'tree_models_results/top_20_stable_structures.csv',
        'tree_models_results/top_20_stable_structures_summary.csv'
    ]
    
    # Check priority files first
    for filepath in priority_files:
        if Path(filepath).exists():
            df = pd.read_csv(filepath)
            
            # For tree models results, use the first row (best structure)
            if len(df) > 0:
                best_row = df.iloc[0]
                best_structure = best_row.get('structure_id', 'structure_637')
                
                # Get energy from available columns
                energy_cols = ['predicted_energy', 'xgboost_prediction', 'actual_energy', 'energy']
                best_energy = None
                for col in energy_cols:
                    if col in best_row:
                        best_energy = best_row[col]
                        break
                
                if best_energy is None:
                    best_energy = -1555.588  # Default for structure_637
                
                print(f"✅ Found reference structure: {best_structure} (E = {best_energy:.6f} eV)")
                print(f"📁 Source: {filepath}")
                return filepath, best_structure
    
    # Fallback to other results
    fallback_files = [
        'kernel_models_analysis/top_20_stable_structures.csv',
        'linear_models_results/top_20_stable_structures.csv',
        'task2/top_20_stable_structures.csv'
    ]
    
    for filepath in fallback_files:
        if Path(filepath).exists():
            df = pd.read_csv(filepath)
            if 'structure_id' in df.columns:
                best_structure = df.iloc[0]['structure_id']
                print(f"⚠️ Using fallback: {best_structure} from {filepath}")
                return filepath, best_structure
    
    raise FileNotFoundError("No suitable structure file found")


def create_comprehensive_report(analyzer: PerturbationAnalyzer, 
                              n_atoms: int, 
                              strength: int,
                              agreement_metrics: Dict):
    """Create a comprehensive analysis report"""
    report_file = analyzer.output_dir / f'comprehensive_report_n{n_atoms}_s{strength}.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PERTURBATION ANALYSIS COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Reference Structure: {analyzer.structure_perturbator.structure_id}\n")
        f.write(f"Reference Energy: {analyzer.reference_energy:.6f} eV\n")
        f.write(f"Atoms Perturbed: {n_atoms}\n")
        f.write(f"Perturbation Strength: {strength}/10\n")
        f.write(f"Number of Trials: {len(analyzer.results)}\n")
        f.write(f"Average Bond Length: {analyzer.structure_perturbator.avg_bond_length:.3f} Å\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        model_types = [col.replace('_perturbed_energy', '') 
                      for col in analyzer.results.columns 
                      if col.endswith('_perturbed_energy')]
        
        for model_type in model_types:
            perturbed = analyzer.results[f'{model_type}_perturbed_energy'].dropna()
            base = analyzer.results[f'{model_type}_base_energy'].dropna()
            delta = analyzer.results[f'{model_type}_delta_energy'].dropna()
            sensitivity = analyzer.results[f'{model_type}_sensitivity'].dropna()
            
            if len(perturbed) > 0:
                mae_from_ref = np.mean(np.abs(perturbed - analyzer.reference_energy))
                rmse_from_ref = np.sqrt(np.mean((perturbed - analyzer.reference_energy)**2))
                
                f.write(f"\n{model_type.upper()}:\n")
                f.write(f"  Base Prediction: {base.iloc[0]:.6f} eV\n")
                f.write(f"  Mean Perturbed Energy: {perturbed.mean():.6f} ± {perturbed.std():.6f} eV\n")
                f.write(f"  Energy Change: {delta.mean():.6f} ± {delta.std():.6f} eV\n")
                f.write(f"  Sensitivity: {sensitivity.mean():.3f} ± {sensitivity.std():.3f} eV/Å\n")
                f.write(f"  MAE from Reference: {mae_from_ref:.6f} eV\n")
                f.write(f"  RMSE from Reference: {rmse_from_ref:.6f} eV\n")
        
        # Model Agreement Analysis
        if agreement_metrics:
            f.write(f"\nMODEL AGREEMENT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            if 'correlations' in agreement_metrics:
                f.write("Pairwise Correlations:\n")
                for pair, corr in agreement_metrics['correlations'].items():
                    f.write(f"  {pair}: {corr:.3f}\n")
                f.write(f"Mean Correlation: {agreement_metrics['mean_correlation']:.3f}\n\n")
            
            if 'mean_prediction_variance' in agreement_metrics:
                f.write(f"Mean Prediction Variance: {agreement_metrics['mean_prediction_variance']:.6f}\n")
                f.write(f"High Agreement Trials: {len(agreement_metrics.get('high_agreement_indices', []))}\n")
                f.write(f"Low Agreement Trials: {len(agreement_metrics.get('low_agreement_indices', []))}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION & INSIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("• Lower sensitivity values indicate more robust models\n")
        f.write("• High correlation between models suggests consistent learning\n")
        f.write("• Large energy changes may indicate overfitting or feature instability\n")
        f.write("• Models with similar sensitivity patterns likely learned similar physics\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Detailed Results: perturbation_results_n{n_atoms}_s{strength}.csv\n")
        f.write(f"• Main Plots: perturbation_analysis_n{n_atoms}_s{strength}.png\n")
        f.write(f"• Model Comparison: model_comparison_n{n_atoms}_s{strength}.png\n")
        f.write(f"• This Report: comprehensive_report_n{n_atoms}_s{strength}.txt\n")
    
    print(f"📄 Comprehensive report saved to {report_file}")


def main():
    """Main execution function with enhanced parameter control"""
    parser = argparse.ArgumentParser(
        description="Perturbation Analysis for Au₂₀ Cluster Energy Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python task3.py                                 # Interactive mode (default)
  python task3.py --structure 350.xyz --n-atoms 2 --strength 7 --trials 15
  python task3.py --auto --n-atoms 1 --strength 5 --trials 10
  python task3.py --list-structures               # Show available structures
        """
    )
    
    # Enhanced structure specification
    parser.add_argument('--structure', type=str,
                       help='Specific structure ID (e.g., 350.xyz)')
    parser.add_argument('--structure-file', type=str,
                       help='Path to structure file (CSV or XYZ)')
    parser.add_argument('--structure-id', type=str, default='350.xyz',
                       help='Structure ID to use from CSV file')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically find best structure from previous results')
    parser.add_argument('--list-structures', action='store_true',
                       help='List available structures from Task 2 and exit')
    
    # Enhanced perturbation parameters with better defaults
    parser.add_argument('--n-atoms', type=int, choices=[1, 2, 3], default=1,
                       help='Number of atoms to perturb (1-3) [default: 1]')
    parser.add_argument('--strength', type=int, choices=range(1, 11), default=5,
                       help='Perturbation strength (1-10) [default: 5]')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of perturbation trials [default: 10]')
    
    # Model specification
    parser.add_argument('--model-dir', type=str,
                       help='Directory containing trained models')
    parser.add_argument('--models', nargs='+',
                       help='Specific model files to load')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./perturbation_results',
                       help='Output directory for results [default: ./perturbation_results]')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    print("🧪 PERTURBATION ANALYSIS FOR Au₂₀ CLUSTERS")
    print("=" * 60)
    
    # Handle special modes
    if args.list_structures:
        list_available_structures_from_task2()
        return 0
    
    # If no arguments provided (or minimal), run interactive mode
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and any(arg in sys.argv for arg in ['--help', '-h'])):
        return run_interactive_mode()
    
    # Check if this is essentially a "default" run (only output dir or simple args)
    has_structure_args = any([args.structure, args.structure_file, args.auto])
    has_param_args = any([
        args.n_atoms != 1, args.strength != 5, args.trials != 10
    ])
    
    if not has_structure_args and not has_param_args and not args.models and not args.model_dir:
        # No significant arguments provided, go interactive
        return run_interactive_mode()
    
    # Initialize analyzer
    analyzer = PerturbationAnalyzer(args.output_dir)
    
    # Enhanced structure loading
    try:
        if args.structure:
            # Use specific structure from Task 2 or data directory
            structure_file, structure_id = load_specific_structure(args.structure)
            analyzer.load_reference_structure(structure_file)
        elif args.auto:
            structure_file, structure_id = load_reference_structure_auto()
            analyzer.load_reference_structure(structure_file)
        elif args.structure_file:
            analyzer.load_reference_structure(args.structure_file)
        else:
            # Try to find automatically
            structure_file, structure_id = load_reference_structure_auto()
            analyzer.load_reference_structure(structure_file)
    except Exception as e:
        print(f"❌ Failed to load structure: {e}")
        return 1
    
    # Load models
    try:
        if args.models:
            # Use specified models
            model_paths = {f'model_{i}': path for i, path in enumerate(args.models)}
        elif args.model_dir:
            # Search specific directory
            model_dir = Path(args.model_dir)
            model_paths = {}
            for model_file in model_dir.glob('*.joblib'):
                model_paths[model_file.stem] = str(model_file)
        else:
            # Auto-discover models
            model_paths = find_available_models()
        
        if not model_paths:
            print("❌ No models found! Please specify model files or directories.")
            return 1
        
        print(f"📁 Found {len(model_paths)} models:")
        for name, path in model_paths.items():
            print(f"   {name}: {path}")
        
        analyzer.evaluator.load_models(model_paths)
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Enhanced parameter display
    print(f"\n⚙️ ANALYSIS PARAMETERS:")
    print(f"   Structure: {analyzer.structure_perturbator.structure_id}")
    print(f"   Atoms to perturb: {args.n_atoms}")
    print(f"   Perturbation strength: {args.strength}/10")
    print(f"   Number of trials: {args.trials}")
    print(f"   Output directory: {args.output_dir}")
    
    # Run analysis
    try:
        analyzer.run_perturbation_analysis(
            n_atoms_to_perturb=args.n_atoms,
            perturbation_strength=args.strength,
            n_trials=args.trials
        )
        
        # Calculate model agreement
        agreement_metrics = analyzer.calculate_model_agreement()
        
        # Create comprehensive report
        create_comprehensive_report(analyzer, args.n_atoms, args.strength, agreement_metrics)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📂 Results saved to: {analyzer.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def list_available_structures_from_task2():
    """List structures available from Task 2 analysis"""
    print("📋 AVAILABLE STRUCTURES FROM TASK 2:")
    print("=" * 60)
    
    try:
        task2_dir = Path("task2")
        top_10_path = task2_dir / "top_10_ranking_table.csv"
        
        if top_10_path.exists():
            df = pd.read_csv(top_10_path)
            print("🏆 TOP 10 STRUCTURES:")
            for _, row in df.iterrows():
                print(f"   {row['Rank']:2d}. {row['Structure_ID']} - "
                      f"E = {row['Energy_eV']:.3f} eV - "
                      f"{row['Shape_Description']} ({row['Family']})")
        else:
            print("⚠️ Task 2 results not found")
            
        # Also show tree models results
        tree_path = Path("tree_models_results/top_20_stable_structures.csv")
        if tree_path.exists():
            print("\n🌳 TREE MODELS TOP STRUCTURES:")
            tree_df = pd.read_csv(tree_path)
            for i in range(min(5, len(tree_df))):
                row = tree_df.iloc[i]
                print(f"   {i+1:2d}. {row['structure_id']} - "
                      f"E = {row.get('predicted_energy', row.get('xgboost_prediction', 'N/A')):.3f} eV")
        
    except Exception as e:
        print(f"❌ Error loading structure list: {e}")


def load_specific_structure(structure_id: str) -> Tuple[str, str]:
    """Load a specific structure by ID"""
    if not structure_id.endswith('.xyz'):
        structure_id += '.xyz'
    
    # First try Task 2 elite dataset
    task2_dir = Path("task2")
    elite_path = task2_dir / "dataset_elite.csv"
    
    if elite_path.exists():
        df = pd.read_csv(elite_path)
        if structure_id in df['structure_id'].values:
            print(f"✅ Found {structure_id} in Task 2 elite dataset")
            return str(elite_path), structure_id
    
    # Try tree models results
    tree_path = Path("tree_models_results/top_20_stable_structures.csv")
    if tree_path.exists():
        df = pd.read_csv(tree_path)
        if structure_id in df['structure_id'].values:
            print(f"✅ Found {structure_id} in tree models results")
            return str(tree_path), structure_id
    
    # Try original data directory
    data_file = Path(f"data/Au20_OPT_1000/{structure_id}")
    if data_file.exists():
        print(f"✅ Found {structure_id} in original data")
        return str(data_file), structure_id
    
    raise FileNotFoundError(f"Structure {structure_id} not found in any dataset")


def get_user_input_int(prompt: str, min_val: int, max_val: int, default: int = None) -> int:
    """Get integer input from user with validation"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} [default: {default}]: ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n🚫 Cancelled by user")
            exit(0)


def select_task_results() -> Tuple[str, str]:
    """Interactive selection of which task results to use"""
    print("\n📋 SELECT ANALYSIS RESULTS SOURCE:")
    print("=" * 50)
    print("1. XGBoost Model - Advanced ML (Best performing tree model)")
    print("2. Task 2 Analysis - Structure Analysis from task2.py results")
    
    while True:
        try:
            choice = input("\nWhich results would you like to use? (1/2): ").strip()
            
            if choice == "1":
                # Check if XGBoost model exists
                xgboost_path = Path("tree_models_results/trained_models/xgboost_model.joblib")
                if xgboost_path.exists():
                    return "xgboost", str(xgboost_path.parent.parent / "top_20_stable_structures.csv")
                else:
                    print("❌ XGBoost model not found!")
                    continue
                    
            elif choice == "2":
                # Check for Task 2 results - use descriptors.csv which has all features
                descriptors_path = Path("au_cluster_analysis_results/descriptors.csv")
                top_10_path = Path("task2/top_10_ranking_table.csv")
                
                if descriptors_path.exists() and top_10_path.exists():
                    return "task2", str(descriptors_path)
                else:
                    print("❌ Task 2 results not found! Need both descriptors.csv and top_10_ranking_table.csv")
                    continue
            else:
                print("❌ Please enter 1 or 2")
                
        except KeyboardInterrupt:
            print("\n🚫 Cancelled by user")
            exit(0)


def load_coordinates_from_xyz(structure_id: str) -> np.ndarray:
    """Load coordinates from original XYZ file"""
    if not structure_id.endswith('.xyz'):
        structure_id += '.xyz'
    
    # Try original data directory
    xyz_file = Path(f"data/Au20_OPT_1000/{structure_id}")
    
    if not xyz_file.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    coordinates = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        coordinates.append([x, y, z])
    
    return np.array(coordinates)


def list_and_select_structure(results_file: str, task_type: str) -> str:
    """List available structures and let user select one"""
    try:
        if task_type == "task2":
            # For Task 2, load the top 10 ranking table and descriptors
            top_10_df = pd.read_csv("task2/top_10_ranking_table.csv")
            descriptors_df = pd.read_csv(results_file)  # results_file is descriptors.csv
            
            print(f"\n📊 AVAILABLE STRUCTURES FROM TASK 2 ANALYSIS:")
            print("=" * 70)
            print("Rank | Structure ID | Energy (eV) | Shape Description")
            print("-" * 70)
            
            for i, row in top_10_df.iterrows():
                structure_id = row['Structure_ID']
                energy = row['Energy_eV']
                shape = row['Shape_Description']
                print(f"{i+1:4d} | {structure_id:12s} | {energy:10.3f} | {shape}")
            
            print(f"\nShowing top 10 structures from Task 2 analysis")
            
        elif task_type == "xgboost":
            # For XGBoost, show the tree model results
            df = pd.read_csv(results_file)
            
            print(f"\n📊 AVAILABLE STRUCTURES FROM XGBOOST PREDICTIONS:")
            print("=" * 60)
            print("Rank | Structure ID | Predicted Energy (eV)")
            print("-" * 45)
            
            for i, row in df.head(15).iterrows():
                energy_val = row.get('predicted_energy', row.get('xgboost_prediction', 0.0))
                print(f"{i+1:4d} | {row['structure_id']:12s} | {energy_val:.6f}")
            
            print(f"\nShowing top 15 structures. Total available: {len(df)}")
        
        # Let user select structure
        while True:
            try:
                print(f"\nSelect structure:")
                if task_type == "task2":
                    print("• Enter rank number (1-10) for quick selection")
                    max_rank = 10
                    df_to_use = top_10_df
                    id_column = 'Structure_ID'
                else:
                    print("• Enter rank number (1-15) for quick selection")
                    max_rank = 15
                    df_to_use = df
                    id_column = 'structure_id'
                
                print("• Enter specific structure ID (e.g., '350.xyz')")
                print("• Enter 'a' for automatic best selection")
                print("• Enter 'all' to run analysis on all structures")
                
                choice = input("Your choice: ").strip()
                
                if choice.lower() == 'a':
                    # Auto-select best (first) structure
                    best_structure = df_to_use.iloc[0][id_column]
                    if task_type == "task2":
                        energy = df_to_use.iloc[0]['Energy_eV']
                    else:
                        energy_col = None
                        for col in ['predicted_energy', 'xgboost_prediction', 'energy']:
                            if col in df_to_use.columns:
                                energy_col = col
                                break
                        energy = df_to_use.iloc[0][energy_col] if energy_col else 0.0
                    
                    print(f"✅ Auto-selected: {best_structure} (E = {energy:.6f} eV)")
                    return best_structure
                
                elif choice.lower() == 'all':
                    # Return special indicator for all structures
                    print(f"✅ Selected: ALL structures ({len(df_to_use)} total)")
                    return "ALL_STRUCTURES"
                
                elif choice.isdigit():
                    rank = int(choice)
                    if 1 <= rank <= min(max_rank, len(df_to_use)):
                        selected_structure = df_to_use.iloc[rank-1][id_column]
                        print(f"✅ Selected: {selected_structure}")
                        return selected_structure
                    else:
                        print(f"❌ Please enter a rank between 1 and {min(max_rank, len(df_to_use))}")
                
                elif choice.endswith('.xyz') or choice in df_to_use[id_column].values:
                    # Direct structure ID
                    if not choice.endswith('.xyz'):
                        choice += '.xyz'
                    if choice in df_to_use[id_column].values:
                        print(f"✅ Selected: {choice}")
                        return choice
                    else:
                        print(f"❌ Structure {choice} not found in dataset")
                
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except ValueError:
                print("❌ Invalid input. Please try again.")
            except KeyboardInterrupt:
                print("\n🚫 Cancelled by user")
                exit(0)
                
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        exit(1)


def get_perturbation_parameters() -> Tuple[int, int, int]:
    """Interactive input for perturbation parameters"""
    print(f"\n⚙️ PERTURBATION PARAMETERS:")
    print("=" * 50)
    
    # Number of atoms to perturb
    print("\n🔸 Number of atoms to perturb:")
    print("  1 = Single atom displacement (minimal disturbance)")
    print("  2 = Two-atom displacement (moderate structural change)")  
    print("  3 = Three-atom displacement (significant structural change)")
    
    n_atoms = get_user_input_int(
        "How many atoms to perturb? (1-3)", 
        min_val=1, max_val=3, default=1
    )
    
    # Perturbation strength
    print(f"\n🔸 Perturbation strength:")
    print("  1-3  = Weak (0.01-0.08 Å) - Small thermal fluctuations")
    print("  4-6  = Medium (0.10-0.18 Å) - Moderate structural deformation")
    print("  7-10 = Strong (0.21-0.30 Å) - Large structural changes")
    
    strength = get_user_input_int(
        "Perturbation strength (1-10)", 
        min_val=1, max_val=10, default=5
    )
    
    # Number of trials
    print(f"\n🔸 Number of trials:")
    print("  5-10   = Quick analysis")
    print("  15-25  = Standard analysis") 
    print("  30-50  = Comprehensive analysis")
    
    trials = get_user_input_int(
        "Number of perturbation trials (5-100)", 
        min_val=5, max_val=100, default=15
    )
    
    return n_atoms, strength, trials


def get_comprehensive_perturbation_parameters() -> Tuple[List[int], List[int], int]:
    """Get parameters for comprehensive analysis across all atom counts and strength categories"""
    print(f"\n⚙️ COMPREHENSIVE PERTURBATION PARAMETERS:")
    print("=" * 60)
    
    # All atom counts
    print("\n🔸 Atoms to perturb: ALL (1, 2, 3)")
    print("  1 = Single atom displacement (minimal disturbance)")
    print("  2 = Two-atom displacement (moderate structural change)")  
    print("  3 = Three-atom displacement (significant structural change)")
    n_atoms_list = [1, 2, 3]
    
    # Representative strengths from each category
    print(f"\n🔸 Perturbation strengths: REPRESENTATIVE FROM EACH CATEGORY")
    print("  2 = Weak category (1-3): Small thermal fluctuations")
    print("  5 = Medium category (4-6): Moderate structural deformation")
    print("  9 = Strong category (7-10): Large structural changes")
    strength_list = [2, 5, 9]  # Representative from each category
    
    # Number of trials
    print(f"\n🔸 Number of trials:")
    print("  5-10   = Quick analysis")
    print("  15-25  = Standard analysis") 
    print("  30-50  = Comprehensive analysis")
    
    trials = get_user_input_int(
        "Number of perturbation trials per combination (5-100)", 
        min_val=5, max_val=100, default=15
    )
    
    print(f"\n📊 ANALYSIS SCOPE:")
    print(f"   • {len(n_atoms_list)} atom counts: {n_atoms_list}")
    print(f"   • {len(strength_list)} strength levels: {strength_list}")
    print(f"   • {trials} trials each")
    print(f"   • Total combinations: {len(n_atoms_list) * len(strength_list)} = {len(n_atoms_list) * len(strength_list)} analyses")
    
    return n_atoms_list, strength_list, trials


def run_all_structures_analysis(task_type: str, results_file: str, n_atoms: int, strength: int, trials: int) -> int:
    """Run perturbation analysis on all available structures"""
    print(f"\n🚀 RUNNING ANALYSIS ON ALL STRUCTURES...")
    print("=" * 60)
    
    try:
        # Get list of all structures
        if task_type == "task2":
            top_10_df = pd.read_csv("task2/top_10_ranking_table.csv")
            structure_list = top_10_df['Structure_ID'].tolist()
            print(f"📊 Processing {len(structure_list)} structures from Task 2")
        else:
            df = pd.read_csv(results_file)
            structure_list = df['structure_id'].head(15).tolist()  # Top 15 for XGBoost
            print(f"📊 Processing {len(structure_list)} structures from XGBoost results")
        
        # Create combined output directory
        combined_output_dir = Path('./perturbation_results_all_structures')
        combined_output_dir.mkdir(exist_ok=True)
        
        all_results = []
        summary_data = []
        
        for i, structure_id in enumerate(structure_list):
            print(f"\n{'='*50}")
            print(f"Processing {i+1}/{len(structure_list)}: {structure_id}")
            print(f"{'='*50}")
            
            try:
                # Initialize analyzer for this structure
                structure_output_dir = combined_output_dir / f"structure_{structure_id.replace('.xyz', '')}"
                analyzer = PerturbationAnalyzer(str(structure_output_dir))
                
                # Load structure
                if task_type == "task2":
                    analyzer.load_reference_structure(results_file, structure_id)
                else:
                    # Create temp CSV for this structure
                    df = pd.read_csv(results_file)
                    selected_row = df[df['structure_id'] == structure_id]
                    if len(selected_row) == 0:
                        print(f"⚠️ Structure {structure_id} not found, skipping...")
                        continue
                    
                    temp_df = selected_row.copy()
                    temp_file = combined_output_dir / f"temp_{structure_id.replace('.', '_')}.csv"
                    temp_df.to_csv(temp_file, index=False)
                    analyzer.load_reference_structure(str(temp_file), structure_id)
                    temp_file.unlink()  # Clean up
                
                # Load models
                if task_type == "xgboost":
                    xgboost_path = Path('tree_models_results/trained_models/xgboost_model.joblib')
                    model_paths = {'xgboost': str(xgboost_path)}
                else:
                    model_paths = find_available_models()
                
                analyzer.evaluator.load_models(model_paths)
                
                # Run analysis
                analyzer.run_perturbation_analysis(
                    n_atoms_to_perturb=n_atoms,
                    perturbation_strength=strength,
                    n_trials=trials
                )
                
                # Collect results for summary
                results_df = analyzer.results
                model_type = list(analyzer.evaluator.models.keys())[0]
                
                # Calculate summary metrics
                perturbed_energies = results_df[f'{model_type}_perturbed_energy'].dropna()
                delta_energies = results_df[f'{model_type}_delta_energy'].dropna()
                sensitivities = results_df[f'{model_type}_sensitivity'].dropna()
                
                if len(perturbed_energies) > 0:
                    summary_data.append({
                        'structure_id': structure_id,
                        'reference_energy': analyzer.reference_energy,
                        'base_prediction': results_df[f'{model_type}_base_energy'].iloc[0],
                        'mean_perturbed_energy': perturbed_energies.mean(),
                        'std_perturbed_energy': perturbed_energies.std(),
                        'mean_energy_change': delta_energies.mean(),
                        'std_energy_change': delta_energies.std(),
                        'mean_sensitivity': sensitivities.mean(),
                        'std_sensitivity': sensitivities.std(),
                        'mae_from_reference': np.mean(np.abs(perturbed_energies - analyzer.reference_energy)),
                        'rmse_from_reference': np.sqrt(np.mean((perturbed_energies - analyzer.reference_energy)**2)),
                        'avg_bond_length': analyzer.structure_perturbator.avg_bond_length,
                        'min_bond_length': analyzer.structure_perturbator.min_bond_length
                    })
                
                print(f"✅ Completed analysis for {structure_id}")
                
            except Exception as e:
                print(f"❌ Failed to analyze {structure_id}: {e}")
                continue
        
        # Create comprehensive summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = combined_output_dir / 'all_structures_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            
            # Create summary visualizations
            create_all_structures_summary_plots(summary_df, combined_output_dir, task_type, n_atoms, strength)
            
            # Create summary report
            create_all_structures_report(summary_df, combined_output_dir, task_type, n_atoms, strength, trials)
            
            # Create representative structure visualizations
            try:
                print(f"\n🎨 CREATING REPRESENTATIVE STRUCTURE VISUALIZATIONS")
                print("=" * 60)
                
                # Create a temporary analyzer for representative visualizations
                repr_analyzer = PerturbationAnalyzer(str(combined_output_dir))
                
                # Select representative structures from the summary data
                representatives = repr_analyzer.select_representative_structures(summary_df)
                
                # Create visualizations for representatives
                visualization_files = repr_analyzer.create_representative_visualizations(
                    representatives, n_atoms=n_atoms, strength=strength
                )
                
                print(f"✅ Representative visualizations created:")
                for category, filename in visualization_files.items():
                    if filename:
                        print(f"   {category}: {filename}")
                    else:
                        print(f"   {category}: Failed")
                        
            except Exception as e:
                print(f"⚠️ Failed to create representative visualizations: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n✅ ALL STRUCTURES ANALYSIS COMPLETED!")
            print(f"📊 Successfully analyzed {len(summary_data)} structures")
            print(f"📂 Results saved to: {combined_output_dir}")
            print(f"📋 Summary file: {summary_file}")
            
            return 0
        else:
            print("❌ No structures were successfully analyzed")
            return 1
            
    except Exception as e:
        print(f"❌ All structures analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_comprehensive_all_structures_analysis(task_type: str, results_file: str, n_atoms_list: List[int], strength_list: List[int], trials: int) -> int:
    """Run comprehensive perturbation analysis on all structures with all parameter combinations"""
    print(f"\n🚀 RUNNING COMPREHENSIVE ANALYSIS ON ALL STRUCTURES...")
    print("=" * 70)
    
    try:
        # Get list of all structures
        if task_type == "task2":
            top_10_df = pd.read_csv("task2/top_10_ranking_table.csv")
            structure_list = top_10_df['Structure_ID'].tolist()
            print(f"📊 Processing {len(structure_list)} structures from Task 2")
        else:
            df = pd.read_csv(results_file)
            structure_list = df['structure_id'].head(15).tolist()  # Top 15 for XGBoost
            print(f"📊 Processing {len(structure_list)} structures from XGBoost results")
        
        # Create combined output directory
        combined_output_dir = Path('./perturbation_results_comprehensive_all_structures')
        combined_output_dir.mkdir(exist_ok=True)
        
        # Track all combinations
        total_combinations = len(structure_list) * len(n_atoms_list) * len(strength_list)
        print(f"📈 Total analyses to run: {total_combinations}")
        print(f"   • {len(structure_list)} structures")
        print(f"   • {len(n_atoms_list)} atom counts: {n_atoms_list}")
        print(f"   • {len(strength_list)} strength levels: {strength_list}")
        print(f"   • {trials} trials per combination")
        
        all_summary_data = []
        completed_analyses = 0
        
        for struct_idx, structure_id in enumerate(structure_list):
            print(f"\n{'='*60}")
            print(f"STRUCTURE {struct_idx+1}/{len(structure_list)}: {structure_id}")
            print(f"{'='*60}")
            
            try:
                # Load structure once for all parameter combinations
                structure_output_dir = combined_output_dir / f"structure_{structure_id.replace('.xyz', '')}"
                structure_output_dir.mkdir(exist_ok=True)
                
                # Load structure data
                if task_type == "task2":
                    # Load descriptors.csv to get structure data
                    df = pd.read_csv(results_file)
                    structure_data = df[df['filename'] == structure_id]
                    if len(structure_data) == 0:
                        print(f"⚠️ Structure {structure_id} not found in descriptors, skipping...")
                        continue
                    reference_energy = structure_data.iloc[0]['energy']
                else:
                    # Load XGBoost results
                    df = pd.read_csv(results_file)
                    structure_data = df[df['structure_id'] == structure_id]
                    if len(structure_data) == 0:
                        print(f"⚠️ Structure {structure_id} not found in XGBoost results, skipping...")
                        continue
                    reference_energy = structure_data.iloc[0].get('predicted_energy', structure_data.iloc[0].get('xgboost_prediction', 0.0))
                
                # Load models once
                if task_type == "xgboost":
                    xgboost_path = Path('tree_models_results/trained_models/xgboost_model.joblib')
                    model_paths = {'xgboost': str(xgboost_path)}
                else:
                    model_paths = find_available_models()
                
                # Run all parameter combinations for this structure
                for n_atoms in n_atoms_list:
                    for strength in strength_list:
                        print(f"\n   📊 Running: {n_atoms} atoms, strength {strength}, {trials} trials")
                        
                        try:
                            # Initialize analyzer for this combination
                            combo_output_dir = structure_output_dir / f"n{n_atoms}_s{strength}"
                            analyzer = PerturbationAnalyzer(str(combo_output_dir))
                            
                            # Load structure
                            if task_type == "task2":
                                analyzer.load_reference_structure(results_file, structure_id)
                            else:
                                # Create temp CSV for this structure
                                temp_df = structure_data.copy()
                                temp_file = combo_output_dir / f"temp_{structure_id.replace('.', '_')}.csv"
                                temp_df.to_csv(temp_file, index=False)
                                analyzer.load_reference_structure(str(temp_file), structure_id)
                                temp_file.unlink()  # Clean up
                            
                            analyzer.evaluator.load_models(model_paths)
                            
                            # Run analysis
                            analyzer.run_perturbation_analysis(
                                n_atoms_to_perturb=n_atoms,
                                perturbation_strength=strength,
                                n_trials=trials
                            )
                            
                            # Collect results for summary
                            results_df = analyzer.results
                            model_type = list(analyzer.evaluator.models.keys())[0]
                            
                            # Calculate summary metrics
                            perturbed_energies = results_df[f'{model_type}_perturbed_energy'].dropna()
                            delta_energies = results_df[f'{model_type}_delta_energy'].dropna()
                            sensitivities = results_df[f'{model_type}_sensitivity'].dropna()
                            
                            if len(perturbed_energies) > 0:
                                all_summary_data.append({
                                    'structure_id': structure_id,
                                    'n_atoms': n_atoms,
                                    'strength': strength,
                                    'trials': trials,
                                    'reference_energy': analyzer.reference_energy,
                                    'base_prediction': results_df[f'{model_type}_base_energy'].iloc[0],
                                    'mean_perturbed_energy': perturbed_energies.mean(),
                                    'std_perturbed_energy': perturbed_energies.std(),
                                    'mean_energy_change': delta_energies.mean(),
                                    'std_energy_change': delta_energies.std(),
                                    'mean_sensitivity': sensitivities.mean(),
                                    'std_sensitivity': sensitivities.std(),
                                    'mae_from_reference': np.mean(np.abs(perturbed_energies - analyzer.reference_energy)),
                                    'rmse_from_reference': np.sqrt(np.mean((perturbed_energies - analyzer.reference_energy)**2)),
                                    'avg_bond_length': analyzer.structure_perturbator.avg_bond_length,
                                    'min_bond_length': analyzer.structure_perturbator.min_bond_length
                                })
                            
                            completed_analyses += 1
                            print(f"      ✅ Completed ({completed_analyses}/{total_combinations})")
                            
                        except Exception as e:
                            print(f"      ❌ Failed combination n{n_atoms}_s{strength}: {e}")
                            continue
                
                print(f"✅ Completed all combinations for {structure_id}")
                
            except Exception as e:
                print(f"❌ Failed structure {structure_id}: {e}")
                continue
        
        # Create comprehensive summary
        if all_summary_data:
            summary_df = pd.DataFrame(all_summary_data)
            summary_file = combined_output_dir / 'comprehensive_all_structures_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            
            # Create comprehensive summary visualizations and reports
            create_comprehensive_summary_analysis(summary_df, combined_output_dir, task_type, n_atoms_list, strength_list, trials)
            
            # Create representative structure visualizations
            try:
                print(f"\n🎨 CREATING REPRESENTATIVE STRUCTURE VISUALIZATIONS")
                print("=" * 60)
                
                # Create a temporary analyzer for representative visualizations
                repr_analyzer = PerturbationAnalyzer(str(combined_output_dir))
                
                # Select representative structures from the summary data
                representatives = repr_analyzer.select_representative_structures(summary_df)
                
                # Create visualizations for representatives using medium parameters (n_atoms=2, strength=5)
                visualization_files = repr_analyzer.create_representative_visualizations(
                    representatives, n_atoms=2, strength=5
                )
                
                print(f"✅ Representative visualizations created:")
                for category, filename in visualization_files.items():
                    if filename:
                        print(f"   {category}: {filename}")
                    else:
                        print(f"   {category}: Failed")
                        
            except Exception as e:
                print(f"⚠️ Failed to create representative visualizations: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n✅ COMPREHENSIVE ALL STRUCTURES ANALYSIS COMPLETED!")
            print(f"📊 Successfully completed {completed_analyses} analyses")
            print(f"📊 Analyzed {len(summary_df['structure_id'].unique())} structures")
            print(f"📂 Results saved to: {combined_output_dir}")
            print(f"📋 Summary file: {summary_file}")
            
            return 0
        else:
            print("❌ No analyses were successfully completed")
            return 1
            
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_comprehensive_summary_analysis(summary_df: pd.DataFrame, output_dir: Path, 
                                        task_type: str, n_atoms_list: List[int], strength_list: List[int], trials: int):
    """Create comprehensive analysis across all parameter combinations"""
    print("📊 Creating comprehensive summary analysis...")
    
    # Create parameter comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    # 1. Sensitivity by atom count
    ax = axes[0, 0]
    for n_atoms in n_atoms_list:
        data = summary_df[summary_df['n_atoms'] == n_atoms]['mean_sensitivity']
        if len(data) > 0:
            ax.hist(data, alpha=0.7, label=f'{n_atoms} atoms', bins=10)
    ax.set_xlabel('Mean Sensitivity (eV/Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sensitivity Distribution by Atom Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sensitivity by strength
    ax = axes[0, 1]
    for strength in strength_list:
        data = summary_df[summary_df['strength'] == strength]['mean_sensitivity']
        if len(data) > 0:
            ax.hist(data, alpha=0.7, label=f'Strength {strength}', bins=10)
    ax.set_xlabel('Mean Sensitivity (eV/Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Sensitivity Distribution by Perturbation Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Energy change by parameters
    ax = axes[0, 2]
    for n_atoms in n_atoms_list:
        for strength in strength_list:
            data = summary_df[(summary_df['n_atoms'] == n_atoms) & (summary_df['strength'] == strength)]
            if len(data) > 0:
                ax.scatter(data['mean_sensitivity'], np.abs(data['mean_energy_change']), 
                          alpha=0.7, label=f'{n_atoms}a-s{strength}', s=30)
    ax.set_xlabel('Mean Sensitivity (eV/Å)')
    ax.set_ylabel('|Mean Energy Change| (eV)')
    ax.set_title('Energy Change vs Sensitivity')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. Boxplot comparison by atom count
    ax = axes[1, 0]
    sensitivity_data = []
    labels = []
    for n_atoms in n_atoms_list:
        data = summary_df[summary_df['n_atoms'] == n_atoms]['mean_sensitivity']
        if len(data) > 0:
            sensitivity_data.append(data.values)
            labels.append(f'{n_atoms} atoms')
    if sensitivity_data:
        bp = ax.boxplot(sensitivity_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    ax.set_ylabel('Mean Sensitivity (eV/Å)')
    ax.set_title('Sensitivity by Atom Count')
    ax.grid(True, alpha=0.3)
    
    # 5. Boxplot comparison by strength
    ax = axes[1, 1]
    sensitivity_data = []
    labels = []
    for strength in strength_list:
        data = summary_df[summary_df['strength'] == strength]['mean_sensitivity']
        if len(data) > 0:
            sensitivity_data.append(data.values)
            labels.append(f'Strength {strength}')
    if sensitivity_data:
        bp = ax.boxplot(sensitivity_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    ax.set_ylabel('Mean Sensitivity (eV/Å)')
    ax.set_title('Sensitivity by Perturbation Strength')
    ax.grid(True, alpha=0.3)
    
    # 6. Prediction error analysis
    ax = axes[1, 2]
    for n_atoms in n_atoms_list:
        data = summary_df[summary_df['n_atoms'] == n_atoms]
        if len(data) > 0:
            errors = np.abs(data['base_prediction'] - data['reference_energy'])
            ax.scatter(data['reference_energy'], errors, alpha=0.7, 
                      label=f'{n_atoms} atoms', s=30)
    ax.set_xlabel('Reference Energy (eV)')
    ax.set_ylabel('Prediction Error (eV)')
    ax.set_title('Prediction Error vs Reference Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Heatmap of mean sensitivity
    ax = axes[2, 0]
    pivot_data = summary_df.groupby(['n_atoms', 'strength'])['mean_sensitivity'].mean().unstack()
    if not pivot_data.empty:
        im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(strength_list)))
        ax.set_yticks(range(len(n_atoms_list)))
        ax.set_xticklabels([f'S{s}' for s in strength_list])
        ax.set_yticklabels([f'{n}a' for n in n_atoms_list])
        ax.set_xlabel('Perturbation Strength')
        ax.set_ylabel('Atoms Perturbed')
        ax.set_title('Mean Sensitivity Heatmap')
        plt.colorbar(im, ax=ax, label='Mean Sensitivity (eV/Å)')
    
    # 8. Structural correlation
    ax = axes[2, 1]
    ax.scatter(summary_df['min_bond_length'], summary_df['mean_sensitivity'], 
               c=summary_df['n_atoms'], alpha=0.7, s=30, cmap='viridis')
    ax.set_xlabel('Minimum Bond Length (Å)')
    ax.set_ylabel('Mean Sensitivity (eV/Å)')
    ax.set_title('Sensitivity vs Structure Properties')
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Atoms Perturbed')
    ax.grid(True, alpha=0.3)
    
    # 9. Energy stability ranking
    ax = axes[2, 2]
    mean_by_structure = summary_df.groupby('structure_id')['mean_sensitivity'].mean().sort_values()
    top_stable = mean_by_structure.head(10)
    bars = ax.bar(range(len(top_stable)), top_stable.values, alpha=0.7)
    ax.set_xlabel('Structure Rank')
    ax.set_ylabel('Mean Sensitivity (eV/Å)')
    ax.set_title('Top 10 Most Stable Structures (Lowest Sensitivity)')
    ax.set_xticks(range(len(top_stable)))
    ax.set_xticklabels([s.replace('.xyz', '') for s in top_stable.index], 
                       rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive plot
    plot_file = output_dir / f'comprehensive_analysis_{task_type}_all_combinations.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comprehensive report
    create_comprehensive_analysis_report(summary_df, output_dir, task_type, n_atoms_list, strength_list, trials)
    
    print(f"📊 Comprehensive analysis saved to: {plot_file}")


def create_comprehensive_analysis_report(summary_df: pd.DataFrame, output_dir: Path, 
                                       task_type: str, n_atoms_list: List[int], strength_list: List[int], trials: int):
    """Create comprehensive analysis report"""
    report_file = output_dir / f'comprehensive_analysis_report_{task_type}.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE PERTURBATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Source: {task_type.upper()}\n")
        f.write(f"Structures Analyzed: {len(summary_df['structure_id'].unique())}\n")
        f.write(f"Atom Counts: {n_atoms_list}\n")
        f.write(f"Perturbation Strengths: {strength_list}\n")
        f.write(f"Trials per Combination: {trials}\n")
        f.write(f"Total Analyses: {len(summary_df)}\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Energy Range: {summary_df['reference_energy'].min():.6f} to {summary_df['reference_energy'].max():.6f} eV\n")
        f.write(f"Overall Mean Sensitivity: {summary_df['mean_sensitivity'].mean():.3f} ± {summary_df['mean_sensitivity'].std():.3f} eV/Å\n")
        f.write(f"Overall Mean Prediction Error: {np.mean(np.abs(summary_df['base_prediction'] - summary_df['reference_energy'])):.6f} eV\n\n")
        
        # Parameter Analysis
        f.write("PARAMETER ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        f.write("By Atom Count:\n")
        for n_atoms in n_atoms_list:
            data = summary_df[summary_df['n_atoms'] == n_atoms]
            if len(data) > 0:
                f.write(f"  {n_atoms} atoms: {data['mean_sensitivity'].mean():.3f} ± {data['mean_sensitivity'].std():.3f} eV/Å "
                       f"({len(data)} analyses)\n")
        
        f.write(f"\nBy Perturbation Strength:\n")
        for strength in strength_list:
            data = summary_df[summary_df['strength'] == strength]
            if len(data) > 0:
                strength_category = "Weak" if strength <= 3 else ("Medium" if strength <= 6 else "Strong")
                f.write(f"  Strength {strength} ({strength_category}): {data['mean_sensitivity'].mean():.3f} ± {data['mean_sensitivity'].std():.3f} eV/Å "
                       f"({len(data)} analyses)\n")
        
        # Top performing structures
        f.write(f"\nTOP 5 MOST STABLE STRUCTURES (Lowest Mean Sensitivity)\n")
        f.write("-" * 40 + "\n")
        mean_by_structure = summary_df.groupby('structure_id')['mean_sensitivity'].mean().sort_values()
        for i, (structure_id, sensitivity) in enumerate(mean_by_structure.head(5).items(), 1):
            f.write(f"{i}. {structure_id:12s} - Mean Sensitivity: {sensitivity:.3f} eV/Å\n")
        
        # Analysis insights
        f.write(f"\nCOMPREHENSIVE ANALYSIS INSIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("• Weak perturbations (strength 2): Probe thermal stability\n")
        f.write("• Medium perturbations (strength 5): Test structural robustness\n")
        f.write("• Strong perturbations (strength 9): Evaluate major deformation response\n")
        f.write("• Multiple atom perturbations reveal collective effects\n")
        f.write("• Sensitivity patterns help identify most stable configurations\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("• Individual combination results in structure subfolders\n")
        f.write("• comprehensive_all_structures_summary.csv - Complete dataset\n")
        f.write(f"• comprehensive_analysis_{task_type}_all_combinations.png - Visual analysis\n")
        f.write(f"• This report: comprehensive_analysis_report_{task_type}.txt\n")
    
    print(f"📄 Comprehensive report saved to: {report_file}")


def create_all_structures_summary_plots(summary_df: pd.DataFrame, output_dir: Path, 
                                       task_type: str, n_atoms: int, strength: int):
    """Create summary plots for all structures analysis"""
    print("📊 Creating summary visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Energy predictions vs reference
    ax = axes[0, 0]
    ax.scatter(summary_df['reference_energy'], summary_df['base_prediction'], alpha=0.7, s=50)
    ax.plot([summary_df['reference_energy'].min(), summary_df['reference_energy'].max()], 
            [summary_df['reference_energy'].min(), summary_df['reference_energy'].max()], 
            'r--', alpha=0.5)
    ax.set_xlabel('Reference Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title('Model Predictions vs Reference')
    ax.grid(True, alpha=0.3)
    
    # 2. Sensitivity distribution
    ax = axes[0, 1]
    ax.hist(summary_df['mean_sensitivity'], bins=15, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Sensitivity (eV/Å)')
    ax.set_ylabel('Number of Structures')
    ax.set_title('Sensitivity Distribution Across Structures')
    ax.grid(True, alpha=0.3)
    
    # 3. Energy change magnitude
    ax = axes[0, 2]
    ax.scatter(summary_df['avg_bond_length'], np.abs(summary_df['mean_energy_change']), 
               alpha=0.7, s=50)
    ax.set_xlabel('Average Bond Length (Å)')
    ax.set_ylabel('|Mean Energy Change| (eV)')
    ax.set_title('Energy Change vs Bond Length')
    ax.grid(True, alpha=0.3)
    
    # 4. Prediction error vs reference energy
    ax = axes[1, 0]
    errors = np.abs(summary_df['base_prediction'] - summary_df['reference_energy'])
    ax.scatter(summary_df['reference_energy'], errors, alpha=0.7, s=50)
    ax.set_xlabel('Reference Energy (eV)')
    ax.set_ylabel('Prediction Error (eV)')
    ax.set_title('Prediction Error vs Reference Energy')
    ax.grid(True, alpha=0.3)
    
    # 5. Sensitivity vs structural properties
    ax = axes[1, 1]
    ax.scatter(summary_df['min_bond_length'], summary_df['mean_sensitivity'], 
               alpha=0.7, s=50)
    ax.set_xlabel('Minimum Bond Length (Å)')
    ax.set_ylabel('Mean Sensitivity (eV/Å)')
    ax.set_title('Sensitivity vs Minimum Bond Length')
    ax.grid(True, alpha=0.3)
    
    # 6. Structure ranking by stability
    ax = axes[1, 2]
    top_10 = summary_df.nsmallest(10, 'reference_energy')
    bars = ax.bar(range(len(top_10)), top_10['reference_energy'], alpha=0.7)
    ax.set_xlabel('Structure Rank')
    ax.set_ylabel('Reference Energy (eV)')
    ax.set_title('Top 10 Most Stable Structures')
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels([s.replace('.xyz', '') for s in top_10['structure_id']], 
                       rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with task type in filename
    plot_file = output_dir / f'all_structures_summary_{task_type}_n{n_atoms}_s{strength}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary plots saved to: {plot_file}")


def create_all_structures_report(summary_df: pd.DataFrame, output_dir: Path, 
                                task_type: str, n_atoms: int, strength: int, trials: int):
    """Create comprehensive report for all structures analysis"""
    report_file = output_dir / f'all_structures_comprehensive_report_{task_type}_n{n_atoms}_s{strength}.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL STRUCTURES PERTURBATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Source: {task_type.upper()}\n")
        f.write(f"Number of Structures Analyzed: {len(summary_df)}\n")
        f.write(f"Atoms Perturbed: {n_atoms}\n")
        f.write(f"Perturbation Strength: {strength}/10\n")
        f.write(f"Trials per Structure: {trials}\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Energy Range: {summary_df['reference_energy'].min():.6f} to {summary_df['reference_energy'].max():.6f} eV\n")
        f.write(f"Mean Sensitivity: {summary_df['mean_sensitivity'].mean():.3f} ± {summary_df['mean_sensitivity'].std():.3f} eV/Å\n")
        f.write(f"Mean Prediction Error: {np.mean(np.abs(summary_df['base_prediction'] - summary_df['reference_energy'])):.6f} eV\n")
        f.write(f"Bond Length Range: {summary_df['avg_bond_length'].min():.3f} to {summary_df['avg_bond_length'].max():.3f} Å\n\n")
        
        # Top structures by stability
        f.write("TOP 5 MOST STABLE STRUCTURES\n")
        f.write("-" * 40 + "\n")
        top_5 = summary_df.nsmallest(5, 'reference_energy')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            f.write(f"{i}. {row['structure_id']:12s} - E = {row['reference_energy']:10.6f} eV "
                   f"(Sensitivity: {row['mean_sensitivity']:6.3f} eV/Å)\n")
        
        # Most sensitive structures
        f.write(f"\nMOST SENSITIVE STRUCTURES\n")
        f.write("-" * 40 + "\n")
        most_sensitive = summary_df.nlargest(5, 'mean_sensitivity')
        for i, (_, row) in enumerate(most_sensitive.iterrows(), 1):
            f.write(f"{i}. {row['structure_id']:12s} - Sensitivity = {row['mean_sensitivity']:6.3f} eV/Å "
                   f"(E = {row['reference_energy']:10.6f} eV)\n")
        
        # Least sensitive structures
        f.write(f"\nLEAST SENSITIVE STRUCTURES\n")
        f.write("-" * 40 + "\n")
        least_sensitive = summary_df.nsmallest(5, 'mean_sensitivity')
        for i, (_, row) in enumerate(least_sensitive.iterrows(), 1):
            f.write(f"{i}. {row['structure_id']:12s} - Sensitivity = {row['mean_sensitivity']:6.3f} eV/Å "
                   f"(E = {row['reference_energy']:10.6f} eV)\n")
        
        # Analysis insights
        f.write(f"\nANALYSIS INSIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("• Lower sensitivity indicates more robust/stable structures\n")
        f.write("• Sensitivity may correlate with structural features like bond lengths\n")
        f.write("• Energy prediction accuracy varies across different structure types\n")
        f.write(f"• Analysis covers {len(summary_df)} structures with {trials} perturbations each\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("• Individual structure results in separate folders\n")
        f.write("• all_structures_summary.csv - Comprehensive summary data\n")
        f.write(f"• all_structures_summary_{task_type}_n{n_atoms}_s{strength}.png - Summary visualizations\n")
        f.write(f"• This report: all_structures_comprehensive_report_{task_type}_n{n_atoms}_s{strength}.txt\n")
    
    print(f"📄 Comprehensive report saved to: {report_file}")


def run_interactive_mode():
    """Enhanced interactive mode for perturbation analysis"""
    print("🎮 INTERACTIVE PERTURBATION ANALYSIS")
    print("=" * 60)
    print("This tool will guide you through setting up a perturbation")
    print("analysis for Au₂₀ cluster energy prediction models.")
    
    try:
        # Step 1: Select task results
        task_type, results_file = select_task_results()
        print(f"✅ Using {task_type} results from: {Path(results_file).name}")
        
        # Step 2: Select structure
        structure_id = list_and_select_structure(results_file, task_type)
        
        # Check if user wants to run on all structures
        if structure_id == "ALL_STRUCTURES":
            # Step 3: Get comprehensive perturbation parameters for all structures
            n_atoms_list, strength_list, trials = get_comprehensive_perturbation_parameters()
            return run_comprehensive_all_structures_analysis(task_type, results_file, n_atoms_list, strength_list, trials)
        
        # Step 3: Get perturbation parameters for single structure
        n_atoms, strength, trials = get_perturbation_parameters()
        
        # Step 4: Confirm and run analysis
        print(f"\n🎯 ANALYSIS CONFIGURATION:")
        print("=" * 50)
        print(f"Result source: {task_type.upper()}")
        print(f"Structure: {structure_id}")
        print(f"Atoms to perturb: {n_atoms}")
        print(f"Perturbation strength: {strength}/10")
        print(f"Number of trials: {trials}")
        
        confirm = input("\nProceed with analysis? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("🚫 Analysis cancelled")
            return 0
        
        # Step 5: Run the analysis
        print(f"\n🚀 STARTING ANALYSIS...")
        print("=" * 60)
        
        # Initialize analyzer
        analyzer = PerturbationAnalyzer('./perturbation_results')
        
        # Load structure
        if task_type == "task2":
            # For Task 2, we need to load from descriptors.csv but get coordinates from XYZ files
            analyzer.load_reference_structure(results_file, structure_id)
        else:
            # For XGBoost, load from the results CSV 
            if results_file.endswith('.csv'):
                # Create a temporary modified CSV with only the selected structure for compatibility
                df = pd.read_csv(results_file)
                selected_row = df[df['structure_id'] == structure_id]
                if len(selected_row) == 0:
                    print(f"❌ Structure {structure_id} not found in {results_file}")
                    return 1
                
                # Use the selected structure
                temp_df = selected_row.copy()
                temp_file = Path(f"temp_structure_{structure_id.replace('.', '_')}.csv")
                temp_df.to_csv(temp_file, index=False)
                
                analyzer.load_reference_structure(str(temp_file), structure_id)
                
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()
            else:
                analyzer.load_reference_structure(results_file, structure_id)
        
        # Load models
        if task_type == "xgboost":
            # For XGBoost option, only load XGBoost model
            xgboost_path = Path('tree_models_results/trained_models/xgboost_model.joblib')
            if xgboost_path.exists():
                model_paths = {'xgboost': str(xgboost_path)}
            else:
                print("❌ XGBoost model not found!")
                return 1
        else:
            # For Task 2, still use XGBoost model for predictions but acknowledge it's Task 2 data
            model_paths = find_available_models()
            if not model_paths:
                print("❌ No models found!")
                return 1
        
        analyzer.evaluator.load_models(model_paths)
        
        # Run analysis
        analyzer.run_perturbation_analysis(
            n_atoms_to_perturb=n_atoms,
            perturbation_strength=strength,
            n_trials=trials
        )
        
        # Calculate agreement metrics
        agreement_metrics = analyzer.calculate_model_agreement()
        
        # Create comprehensive report
        create_comprehensive_report(analyzer, n_atoms, strength, agreement_metrics)
        
        # Create structure visualization for this single analysis
        try:
            print(f"\n🎨 CREATING STRUCTURE VISUALIZATION")
            print("=" * 50)
            
            viz_output_dir = analyzer.output_dir / 'structure_visualization'
            viz_output_dir.mkdir(exist_ok=True)
            
            # Generate one perturbation for visualization (reproducible)
            perturbed_structure, perturbation_info = analyzer.structure_perturbator.perturb_structure(
                n_atoms_to_perturb=n_atoms,
                perturbation_strength=strength,
                seed=42  # Reproducible for documentation
            )
            
            # Create 3D visualization
            filename = analyzer.create_structure_visualization(
                structure_before=analyzer.structure_perturbator.base_structure,
                structure_after=perturbed_structure,
                perturbation_info=perturbation_info,
                structure_id=analyzer.structure_perturbator.structure_id,
                output_dir=viz_output_dir
            )
            
            print(f"✅ Structure visualization saved: {filename}")
            print(f"📁 Location: {viz_output_dir}")
            
        except Exception as e:
            print(f"⚠️ Failed to create structure visualization: {e}")
        
        print(f"\n✅ ANALYSIS COMPLETED!")
        print(f"📂 Results saved to: {analyzer.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🚫 Analysis cancelled by user")
        return 0
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())