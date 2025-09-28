#!/usr/bin/env python3
"""
Task 1: Au Cluster Analysis - ASE-Enhanced Competition Version
Parse xyz files, compute descriptors, and generate statistical summary using ASE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ASE imports
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.geometry.analysis import Analysis
from ase.data import covalent_radii, atomic_numbers
from ase.atoms import Atoms

class AuClusterAnalyzer:
    """ASE-enhanced analyzer for Au cluster xyz files - Competition optimized"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.structures = []
        
    def parse_xyz_file(self, filepath):
        """Parse single xyz file using ASE - robust and accurate"""
        try:
            # Use ASE to read xyz file
            atoms = read(str(filepath))
            
            # Extract energy from comment line (ASE reads it automatically)
            energy = atoms.info.get('energy', None)
            
            # If energy not in info, parse manually from file
            if energy is None:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 1:
                    energy = self._extract_energy(lines[1])
            
            return {
                'filename': filepath.name,
                'n_atoms': len(atoms),
                'energy': energy,
                'atoms': atoms,  # Store ASE Atoms object
                'coords': atoms.get_positions(),
                'elements': atoms.get_chemical_symbols()
            }
            
        except Exception as e:
            print(f"Error parsing {filepath.name}: {e}")
            return None
    
    def _extract_energy(self, energy_line):
        """Extract energy value from comment line"""
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', energy_line)
        for num in numbers:
            try:
                val = float(num)
                # Au cluster energies typically in this range
                if -50000 < val < 50000:
                    return val
            except:
                continue
        return None
    
    def parse_all_files(self):
        """Parse all xyz files in directory"""
        # Check if directory exists
        if not self.data_dir.exists():
            print(f"Directory does not exist: {self.data_dir}")
            return []
            
        # Try different patterns
        xyz_files = list(self.data_dir.glob("*.xyz"))
        if len(xyz_files) == 0:
            xyz_files = list(self.data_dir.glob("**/*.xyz"))
        
        print(f"Found {len(xyz_files)} xyz files")
        if len(xyz_files) == 0:
            print(f"No .xyz files found in: {self.data_dir}")
            print("Contents of directory:")
            try:
                for item in self.data_dir.iterdir():
                    print(f"  {item.name}")
            except:
                print("  Cannot list directory contents")
        
        for filepath in xyz_files:
            structure = self.parse_xyz_file(filepath)
            if structure is not None:
                self.structures.append(structure)
        
        print(f"Successfully parsed {len(self.structures)} files")
        return self.structures
    
    def compute_bond_lengths_ase(self, atoms):
        """Compute bond lengths using ASE neighbor list"""
        # Use natural cutoffs for Au (more accurate)
        cutoffs = natural_cutoffs(atoms, mult=1.2)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        bonds = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if i < j:  # Avoid double counting
                    distance = atoms.get_distance(i, j)
                    bonds.append(distance)
        
        return np.array(bonds)
    
    def compute_coordination_numbers_ase(self, atoms):
        """Compute coordination numbers using ASE"""
        cutoffs = natural_cutoffs(atoms, mult=1.1)  # Standard cutoff
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        coord_nums = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            coord_nums.append(len(indices))
        
        return np.array(coord_nums)
    
    def compute_surface_atoms_ase(self, atoms):
        """Identify surface atoms using coordination analysis"""
        coord_nums = self.compute_coordination_numbers_ase(atoms)
        
        # Surface atoms have lower coordination
        mean_coord = np.mean(coord_nums)
        surface_threshold = mean_coord - 1.0  # Adaptive threshold
        
        surface_atoms = np.sum(coord_nums <= surface_threshold)
        return surface_atoms / len(atoms)
    
    def compute_radius_of_gyration_ase(self, atoms):
        """Compute radius of gyration using ASE"""
        positions = atoms.get_positions()
        center_of_mass = atoms.get_center_of_mass()
        
        distances_sq = np.sum((positions - center_of_mass)**2, axis=1)
        masses = atoms.get_masses()
        
        # Mass-weighted radius of gyration
        rg_sq = np.sum(masses * distances_sq) / np.sum(masses)
        return np.sqrt(rg_sq)
    
    def compute_asphericity_ase(self, atoms):
        """Compute asphericity using ASE"""
        positions = atoms.get_positions()
        com = atoms.get_center_of_mass()
        
        # Center positions
        centered = positions - com
        
        # Gyration tensor
        gyration_tensor = np.dot(centered.T, centered) / len(atoms)
        eigenvals = np.linalg.eigvals(gyration_tensor)
        eigenvals = np.sort(eigenvals)[::-1]
        
        # Asphericity parameter
        if np.sum(eigenvals) > 0:
            asphericity = eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])
            return asphericity / np.sum(eigenvals)
        return 0.0
    
    def compute_advanced_descriptors(self, atoms):
        """Compute advanced structural descriptors"""
        positions = atoms.get_positions()
        
        # Geometric properties
        ranges = np.max(positions, axis=0) - np.min(positions, axis=0)
        
        # Bond angle analysis (simplified)
        bonds = self.compute_bond_lengths_ase(atoms)
        
        # Structural compactness
        volume_estimate = np.prod(ranges)
        compactness = len(atoms) / (volume_estimate + 1e-8)
        
        return {
            'x_range': ranges[0],
            'y_range': ranges[1], 
            'z_range': ranges[2],
            'max_range': np.max(ranges),
            'anisotropy': np.max(ranges) / np.min(ranges) if np.min(ranges) > 0 else 1.0,
            'volume_estimate': volume_estimate,
            'compactness': compactness,
            'bond_variance': np.var(bonds) if len(bonds) > 0 else 0.0
        }
    
    def compute_descriptors(self):
        """Compute all descriptors for parsed structures using ASE"""
        descriptors = []
        
        print(f"Computing ASE-enhanced descriptors for {len(self.structures)} structures...")
        
        for structure in self.structures:
            atoms = structure['atoms']
            
            # Basic properties
            desc = {
                'filename': structure['filename'],
                'n_atoms': structure['n_atoms'],
                'energy': structure['energy'],
                'energy_per_atom': structure['energy'] / structure['n_atoms'] if structure['energy'] else None
            }
            
            # ASE-enhanced bond statistics
            bonds = self.compute_bond_lengths_ase(atoms)  # Fixed: use ASE version
            if len(bonds) > 0:
                desc.update({
                    'mean_bond_length': np.mean(bonds),
                    'std_bond_length': np.std(bonds),
                    'min_bond_length': np.min(bonds),
                    'max_bond_length': np.max(bonds),
                    'n_bonds': len(bonds)
                })
            else:
                desc.update({
                    'mean_bond_length': None,
                    'std_bond_length': None,
                    'min_bond_length': None,
                    'max_bond_length': None,
                    'n_bonds': 0
                })
            
            # ASE-enhanced coordination statistics
            coord_nums = self.compute_coordination_numbers_ase(atoms)  # Fixed: use ASE version
            desc.update({
                'mean_coordination': np.mean(coord_nums),
                'std_coordination': np.std(coord_nums),
                'max_coordination': np.max(coord_nums),
                'min_coordination': np.min(coord_nums)
            })
            
            # ASE-enhanced geometric properties
            desc.update({
                'radius_of_gyration': self.compute_radius_of_gyration_ase(atoms),  # Fixed: use ASE version
                'asphericity': self.compute_asphericity_ase(atoms),  # Fixed: use ASE version
                'surface_fraction': self.compute_surface_atoms_ase(atoms)  # Fixed: use ASE version
            })
            
            # Advanced descriptors
            advanced_desc = self.compute_advanced_descriptors(atoms)
            desc.update(advanced_desc)
            
            descriptors.append(desc)
        
        return pd.DataFrame(descriptors)
    
    def generate_statistics(self, df):
        """Generate statistical summary"""
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        # Dataset overview
        print(f"\nDataset Overview:")
        print(f"  Total structures: {len(df)}")
        print(f"  Structures with energy: {df['energy'].notna().sum()}")
        
        # Size distribution
        print(f"\nSize Distribution:")
        print(f"  Atoms per cluster: {df['n_atoms'].min()} - {df['n_atoms'].max()}")
        print(f"  Average size: {df['n_atoms'].mean():.1f} atoms")
        
        # Energy statistics
        if df['energy'].notna().sum() > 0:
            energies = df['energy'].dropna()
            print(f"\nEnergy Statistics:")
            print(f"  Total energy range: {energies.min():.2f} to {energies.max():.2f} eV")
            print(f"  Average total energy: {energies.mean():.2f} eV")
            
            if df['energy_per_atom'].notna().sum() > 0:
                energy_per_atom = df['energy_per_atom'].dropna()
                print(f"  Energy per atom range: {energy_per_atom.min():.2f} to {energy_per_atom.max():.2f} eV/atom")
                print(f"  Average energy per atom: {energy_per_atom.mean():.2f} eV/atom")
        
        # Structural statistics
        print(f"\nStructural Properties:")
        if df['mean_bond_length'].notna().sum() > 0:
            bond_lengths = df['mean_bond_length'].dropna()
            print(f"  Bond lengths: {bond_lengths.min():.3f} - {bond_lengths.max():.3f} Å (avg: {bond_lengths.mean():.3f})")
        
        coord_nums = df['mean_coordination'].dropna()
        print(f"  Coordination numbers: {coord_nums.min():.1f} - {coord_nums.max():.1f} (avg: {coord_nums.mean():.1f})")
        
        rg_values = df['radius_of_gyration'].dropna()
        print(f"  Radius of gyration: {rg_values.min():.2f} - {rg_values.max():.2f} Å (avg: {rg_values.mean():.2f})")
        
        surface_fractions = df['surface_fraction'].dropna()
        print(f"  Surface fraction: {surface_fractions.min():.2f} - {surface_fractions.max():.2f} (avg: {surface_fractions.mean():.2f})")
        
        return df.describe()
    
    def create_plots(self, df, output_dir=None):
        """Generate basic plots"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Energy vs Size plot
        if df['energy'].notna().sum() > 1:
            plt.figure(figsize=(10, 6))
            valid_data = df.dropna(subset=['energy', 'n_atoms'])
            plt.scatter(valid_data['n_atoms'], valid_data['energy'], alpha=0.7, s=50)
            plt.xlabel('Number of Atoms')
            plt.ylabel('Total Energy (eV)')
            plt.title('Energy vs Cluster Size')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'energy_vs_size.png', dpi=300)
            plt.close()
        
        # 2. Key descriptors distribution
        key_features = ['mean_bond_length', 'mean_coordination', 'radius_of_gyration', 'surface_fraction']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(key_features):
            if feature in df.columns and df[feature].notna().sum() > 0:
                data = df[feature].dropna()
                axes[i].hist(data, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'descriptors_distribution.png', dpi=300)
        plt.close()
        
        # 3. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
            plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def save_results(self, df, output_dir=None):
        """Save results to files"""
        if output_dir is None:
            output_dir = Path('./results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Save descriptors CSV (computed features)
        df.to_csv(output_dir / 'descriptors.csv', index=False)
        
        # Save raw data with coordinates (enhanced with ASE data)
        raw_data = []
        for structure in self.structures:
            atoms = structure['atoms']
            coords = atoms.get_positions()
            elements = atoms.get_chemical_symbols()
            
            for i, (coord, element) in enumerate(zip(coords, elements)):
                raw_data.append({
                    'filename': structure['filename'],
                    'n_atoms': structure['n_atoms'],
                    'energy': structure['energy'],
                    'atom_index': i + 1,
                    'element': element,
                    'x': coord[0],
                    'y': coord[1],
                    'z': coord[2]
                })
        
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_csv(output_dir / 'raw_coordinates.csv', index=False)
        
        # Save summary statistics
        summary_stats = df.describe()
        summary_stats.to_csv(output_dir / 'summary_statistics.csv')
        
        # Save structure summary (one row per file)
        structure_summary = []
        for structure in self.structures:
            structure_summary.append({
                'filename': structure['filename'],
                'n_atoms': structure['n_atoms'],
                'energy': structure['energy'],
                'energy_per_atom': structure['energy'] / structure['n_atoms'] if structure['energy'] else None
            })
        
        summary_df = pd.DataFrame(structure_summary)
        summary_df.to_csv(output_dir / 'structure_summary.csv', index=False)
        
        print(f"Results saved to {output_dir}")
        print(f"  - descriptors.csv: Computed structural features")
        print(f"  - raw_coordinates.csv: All atomic coordinates")  
        print(f"  - structure_summary.csv: Basic info per structure")
        print(f"  - summary_statistics.csv: Statistical summary")
        
        return output_dir
    
    def analyze_feature_target_relationships(self, df, target_col='energy'):
        """Analyze correlations between features and target variable"""
        if target_col not in df.columns or df[target_col].isna().all():
            print(f"Target column '{target_col}' not available for correlation analysis")
            return None
            
        print("\n" + "="*60)
        print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
        print("="*60)
        
        # Select numeric features (excluding target and identifiers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in [target_col, 'energy_per_atom']]
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_col])
        
        correlations = []
        
        for feature in feature_cols:
            if df_clean[feature].notna().sum() > 5:  # Need at least 5 data points
                # Remove missing values for this feature
                mask = df_clean[feature].notna() & df_clean[target_col].notna()
                if mask.sum() < 5:
                    continue
                    
                x = df_clean.loc[mask, feature]
                y = df_clean.loc[mask, target_col]
                
                try:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(x, y)
                    
                    # Spearman correlation  
                    spearman_r, spearman_p = spearmanr(x, y)
                    
                    correlations.append({
                        'feature': feature,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'abs_pearson': abs(pearson_r),
                        'n_samples': mask.sum()
                    })
                except:
                    continue
        
        if not correlations:
            print("No valid correlations found")
            return None
            
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)
        
        print(f"\nTop 10 Features by Correlation with {target_col}:")
        print("-" * 80)
        for _, row in corr_df.head(10).iterrows():
            significance = "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
            print(f"{row['feature']:<25} | Pearson: {row['pearson_r']:.3f}{significance} | Spearman: {row['spearman_r']:.3f} | n={row['n_samples']}")
        
        print("\n** p < 0.01, * p < 0.05")
        
        return corr_df
    
    def create_feature_target_plots(self, df, corr_df, target_col='energy', output_dir=None):
        """Create scatter plots of key features vs target"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Get top 6 features for plotting
        top_features = corr_df.head(6)['feature'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        df_clean = df.dropna(subset=[target_col])
        
        for i, feature in enumerate(top_features):
            if i >= 6:
                break
                
            # Get clean data for this feature
            mask = df_clean[feature].notna() & df_clean[target_col].notna()
            x = df_clean.loc[mask, feature]
            y = df_clean.loc[mask, target_col]
            
            if len(x) < 2:
                continue
            
            # Scatter plot
            axes[i].scatter(x, y, alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)
            
            # Get correlation for this feature
            corr_row = corr_df[corr_df['feature'] == feature].iloc[0]
            r_val = corr_row['pearson_r']
            
            axes[i].set_xlabel(feature.replace('_', ' ').title())
            axes[i].set_ylabel(target_col.replace('_', ' ').title())
            axes[i].set_title(f'{feature}\nr = {r_val:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_target_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature-target relationship plots saved")
    
    def perform_pca_analysis(self, df, output_dir=None):
        """Perform PCA analysis and visualization"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)
        
        # Select numeric features (excluding identifiers and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['energy', 'energy_per_atom']]
        
        # Prepare data
        df_pca = df[feature_cols + ['energy']].dropna()
        
        if len(df_pca) < 5:
            print("Not enough samples for PCA analysis")
            return None
        
        X = df_pca[feature_cols]
        
        # Remove columns with zero variance
        X = X.loc[:, X.var() > 1e-8]
        
        if X.shape[1] < 2:
            print("Not enough features with variance for PCA")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Print explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"Explained Variance by Component:")
        for i in range(min(5, len(explained_var))):
            print(f"  PC{i+1}: {explained_var[i]:.3f} ({explained_var[i]*100:.1f}%)")
        print(f"  Cumulative (first 3): {cumulative_var[2]:.3f} ({cumulative_var[2]*100:.1f}%)")
        
        # Create PCA plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Explained variance plot
        axes[0,0].bar(range(1, min(11, len(explained_var)+1)), explained_var[:10])
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].set_title('PCA Explained Variance')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Cumulative explained variance
        axes[0,1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'bo-')
        axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80%')
        axes[0,1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
        axes[0,1].set_xlabel('Number of Components')
        axes[0,1].set_ylabel('Cumulative Explained Variance')
        axes[0,1].set_title('Cumulative Explained Variance')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. PC1 vs PC2 colored by energy
        if 'energy' in df_pca.columns and not df_pca['energy'].isna().all():
            scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                      c=df_pca['energy'], cmap='viridis', alpha=0.6)
            axes[1,0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[1,0].set_title('PCA: PC1 vs PC2 (colored by energy)')
            plt.colorbar(scatter, ax=axes[1,0], label='Energy')
        else:
            axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
            axes[1,0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[1,0].set_title('PCA: PC1 vs PC2')
        
        # 4. Feature loadings for PC1 and PC2
        loadings = pca.components_[:2].T
        feature_importance = np.abs(loadings).sum(axis=1)
        top_features_idx = np.argsort(feature_importance)[-8:]  # Top 8 features
        
        for i, idx in enumerate(top_features_idx):
            axes[1,1].arrow(0, 0, loadings[idx, 0], loadings[idx, 1], 
                          head_width=0.02, head_length=0.02, fc='red', ec='red')
            axes[1,1].text(loadings[idx, 0]*1.1, loadings[idx, 1]*1.1, 
                         X.columns[idx], fontsize=8)
        
        axes[1,1].set_xlim(-1, 1)
        axes[1,1].set_ylim(-1, 1)
        axes[1,1].set_xlabel('PC1 Loading')
        axes[1,1].set_ylabel('PC2 Loading')
        axes[1,1].set_title('Feature Loadings (PC1 vs PC2)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PCA analysis plots saved")
        
        return pca, X_pca, scaler
    
    def detect_outliers(self, df, output_dir=None):
        """Detect and analyze outliers in the dataset"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("OUTLIER & VARIABILITY ANALYSIS")
        print("="*60)
        
        # Select key features for outlier analysis
        key_features = ['mean_bond_length', 'mean_coordination', 'radius_of_gyration', 
                       'asphericity', 'surface_fraction', 'anisotropy']
        
        available_features = [f for f in key_features if f in df.columns]
        
        outliers_info = []
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:6]):
            data = df[feature].dropna()
            
            if len(data) > 4:
                # Calculate IQR-based outliers
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_indices = df[df[feature].isin(outliers)].index
                
                # Store outlier info
                if len(outliers) > 0:
                    outliers_info.append({
                        'feature': feature,
                        'n_outliers': len(outliers),
                        'outlier_files': df.loc[outlier_indices, 'filename'].tolist()
                    })
                
                # Box plot
                axes[i].boxplot(data, labels=[feature.replace('_', '\n')])
                axes[i].set_title(f'{feature}\n{len(outliers)} outliers')
                axes[i].grid(True, alpha=0.3)
                
                # Highlight outliers
                if len(outliers) > 0:
                    axes[i].scatter([1]*len(outliers), outliers, color='red', s=50, alpha=0.7, label='Outliers')
                    axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print outlier summary
        if outliers_info:
            print("\nOutlier Summary:")
            print("-" * 50)
            for info in outliers_info:
                print(f"{info['feature']:<20} | {info['n_outliers']} outliers")
                if info['n_outliers'] <= 5:  # Show filenames for small number of outliers
                    for filename in info['outlier_files']:
                        print(f"  → {filename}")
        else:
            print("No significant outliers detected using IQR method")
        
        print(f"Outlier analysis plots saved")
        
        return outliers_info
    
    def create_advanced_correlation_heatmap(self, df, output_dir=None):
        """Create enhanced correlation heatmap"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove columns with too many NaN values
        numeric_df = numeric_df.dropna(axis=1, thresh=len(numeric_df)*0.5)
        
        if numeric_df.shape[1] < 2:
            print("Not enough numeric columns for correlation plot")
            return None
        
        plt.figure(figsize=(12, 10))
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Enhanced Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'enhanced_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced correlation heatmap saved")
        
    def run_advanced_analysis(self, df, output_dir=None):
        """Run all advanced analysis features - COMPLETE VERSION"""
        print("\n🔬 Running COMPLETE Advanced Analysis Features...")
        
        if output_dir is None:
            output_dir = Path('./advanced_analysis')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # 1. Feature-Target Relationship Analysis
        print("\n1️⃣ Analyzing Feature-Target Relationships...")
        corr_df = self.analyze_feature_target_relationships(df)
        if corr_df is not None:
            corr_df.to_csv(output_dir / 'feature_correlations.csv', index=False)
            self.create_feature_target_plots(df, corr_df, output_dir=output_dir)
        
        # 2. Enhanced Correlation Heatmap
        print("\n2️⃣ Creating Enhanced Correlation Heatmap...")
        corr_matrix = self.create_advanced_correlation_heatmap(df, output_dir=output_dir)
        
        # 3. PCA Analysis (Dimensionality Reduction)
        print("\n3️⃣ Performing PCA Analysis...")
        pca_results = self.perform_pca_analysis(df, output_dir=output_dir)
        
        # 4. Outlier Detection
        print("\n4️⃣ Detecting Outliers...")
        outliers_info = self.detect_outliers(df, output_dir=output_dir)
        
        # 5. Save outlier info to CSV
        if outliers_info:
            outlier_df = pd.DataFrame(outliers_info)
            outlier_df.to_csv(output_dir / 'outlier_summary.csv', index=False)
        
        print(f"\n✅ COMPLETE Advanced analysis finished! Results saved to: {output_dir}")
        print(f"📊 Generated files:")
        print(f"  - feature_correlations.csv: Feature-energy correlations")
        print(f"  - feature_target_relationships.png: Scatter plots")
        print(f"  - enhanced_correlation_heatmap.png: Full correlation matrix")
        print(f"  - pca_analysis.png: PCA visualizations")
        print(f"  - outlier_analysis.png: Outlier detection plots")
        if outliers_info:
            print(f"  - outlier_summary.csv: Outlier summary table")
        
        return {
            'correlations': corr_df,
            'correlation_matrix': corr_matrix,
            'pca_results': pca_results,
            'outliers': outliers_info
        }
    
    def compute_descriptors(self):
        """Compute all descriptors for parsed structures using ASE"""
        descriptors = []
        
        print(f"Computing ASE-enhanced descriptors for {len(self.structures)} structures...")
        
        for structure in self.structures:
            atoms = structure['atoms']
            
            # Basic properties
            desc = {
                'filename': structure['filename'],
                'n_atoms': structure['n_atoms'],
                'energy': structure['energy'],
                'energy_per_atom': structure['energy'] / structure['n_atoms'] if structure['energy'] else None
            }
            
            # ASE-enhanced bond statistics
            bonds = self.compute_bond_lengths_ase(atoms)  # Fixed: use ASE version
            if len(bonds) > 0:
                desc.update({
                    'mean_bond_length': np.mean(bonds),
                    'std_bond_length': np.std(bonds),
                    'min_bond_length': np.min(bonds),
                    'max_bond_length': np.max(bonds),
                    'n_bonds': len(bonds)
                })
            else:
                desc.update({
                    'mean_bond_length': None,
                    'std_bond_length': None,
                    'min_bond_length': None,
                    'max_bond_length': None,
                    'n_bonds': 0
                })
            
            # ASE-enhanced coordination statistics
            coord_nums = self.compute_coordination_numbers_ase(atoms)  # Fixed: use ASE version
            desc.update({
                'mean_coordination': np.mean(coord_nums),
                'std_coordination': np.std(coord_nums),
                'max_coordination': np.max(coord_nums),
                'min_coordination': np.min(coord_nums)
            })
            
            # ASE-enhanced geometric properties
            desc.update({
                'radius_of_gyration': self.compute_radius_of_gyration_ase(atoms),  # Fixed: use ASE version
                'asphericity': self.compute_asphericity_ase(atoms),  # Fixed: use ASE version
                'surface_fraction': self.compute_surface_atoms_ase(atoms)  # Fixed: use ASE version
            })
            
            # Advanced descriptors
            advanced_desc = self.compute_advanced_descriptors(atoms)
            desc.update(advanced_desc)
            
            descriptors.append(desc)
        
        return pd.DataFrame(descriptors)
    
    def generate_statistics(self, df):
        """Generate statistical summary"""
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        # Dataset overview
        print(f"\nDataset Overview:")
        print(f"  Total structures: {len(df)}")
        print(f"  Structures with energy: {df['energy'].notna().sum()}")
        
        # Size distribution
        print(f"\nSize Distribution:")
        print(f"  Atoms per cluster: {df['n_atoms'].min()} - {df['n_atoms'].max()}")
        print(f"  Average size: {df['n_atoms'].mean():.1f} atoms")
        
        # Energy statistics
        if df['energy'].notna().sum() > 0:
            energies = df['energy'].dropna()
            print(f"\nEnergy Statistics:")
            print(f"  Total energy range: {energies.min():.2f} to {energies.max():.2f} eV")
            print(f"  Average total energy: {energies.mean():.2f} eV")
            
            if df['energy_per_atom'].notna().sum() > 0:
                energy_per_atom = df['energy_per_atom'].dropna()
                print(f"  Energy per atom range: {energy_per_atom.min():.2f} to {energy_per_atom.max():.2f} eV/atom")
                print(f"  Average energy per atom: {energy_per_atom.mean():.2f} eV/atom")
        
        # Structural statistics
        print(f"\nStructural Properties:")
        if df['mean_bond_length'].notna().sum() > 0:
            bond_lengths = df['mean_bond_length'].dropna()
            print(f"  Bond lengths: {bond_lengths.min():.3f} - {bond_lengths.max():.3f} Å (avg: {bond_lengths.mean():.3f})")
        
        coord_nums = df['mean_coordination'].dropna()
        print(f"  Coordination numbers: {coord_nums.min():.1f} - {coord_nums.max():.1f} (avg: {coord_nums.mean():.1f})")
        
        rg_values = df['radius_of_gyration'].dropna()
        print(f"  Radius of gyration: {rg_values.min():.2f} - {rg_values.max():.2f} Å (avg: {rg_values.mean():.2f})")
        
        surface_fractions = df['surface_fraction'].dropna()
        print(f"  Surface fraction: {surface_fractions.min():.2f} - {surface_fractions.max():.2f} (avg: {surface_fractions.mean():.2f})")
        
        return df.describe()
    
    def create_plots(self, df, output_dir=None):
        """Generate basic plots"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Energy vs Size plot
        if df['energy'].notna().sum() > 1:
            plt.figure(figsize=(10, 6))
            valid_data = df.dropna(subset=['energy', 'n_atoms'])
            plt.scatter(valid_data['n_atoms'], valid_data['energy'], alpha=0.7, s=50)
            plt.xlabel('Number of Atoms')
            plt.ylabel('Total Energy (eV)')
            plt.title('Energy vs Cluster Size')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'energy_vs_size.png', dpi=300)
            plt.close()
        
        # 2. Key descriptors distribution
        key_features = ['mean_bond_length', 'mean_coordination', 'radius_of_gyration', 'surface_fraction']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(key_features):
            if feature in df.columns and df[feature].notna().sum() > 0:
                data = df[feature].dropna()
                axes[i].hist(data, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'descriptors_distribution.png', dpi=300)
        plt.close()
        
        # 3. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300)
            plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def save_results(self, df, output_dir=None):
        """Save results to files"""
        if output_dir is None:
            output_dir = Path('./results')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Save descriptors CSV (computed features)
        df.to_csv(output_dir / 'descriptors.csv', index=False)
        
        # Save raw data with coordinates (enhanced with ASE data)
        raw_data = []
        for structure in self.structures:
            atoms = structure['atoms']
            coords = atoms.get_positions()
            elements = atoms.get_chemical_symbols()
            
            for i, (coord, element) in enumerate(zip(coords, elements)):
                raw_data.append({
                    'filename': structure['filename'],
                    'n_atoms': structure['n_atoms'],
                    'energy': structure['energy'],
                    'atom_index': i + 1,
                    'element': element,
                    'x': coord[0],
                    'y': coord[1],
                    'z': coord[2]
                })
        
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_csv(output_dir / 'raw_coordinates.csv', index=False)
        
        # Save summary statistics
        summary_stats = df.describe()
        summary_stats.to_csv(output_dir / 'summary_statistics.csv')
        
        # Save structure summary (one row per file)
        structure_summary = []
        for structure in self.structures:
            structure_summary.append({
                'filename': structure['filename'],
                'n_atoms': structure['n_atoms'],
                'energy': structure['energy'],
                'energy_per_atom': structure['energy'] / structure['n_atoms'] if structure['energy'] else None
            })
        
        summary_df = pd.DataFrame(structure_summary)
        summary_df.to_csv(output_dir / 'structure_summary.csv', index=False)
        
    def analyze_feature_target_relationships(self, df, target_col='energy'):
        """Analyze correlations between features and target variable"""
        if target_col not in df.columns or df[target_col].isna().all():
            print(f"Target column '{target_col}' not available for correlation analysis")
            return None
            
        print("\n" + "="*60)
        print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
        print("="*60)
        
        # Select numeric features (excluding target and identifiers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in [target_col, 'energy_per_atom']]
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_col])
        
        correlations = []
        
        for feature in feature_cols:
            if df_clean[feature].notna().sum() > 5:  # Need at least 5 data points
                # Remove missing values for this feature
                mask = df_clean[feature].notna() & df_clean[target_col].notna()
                if mask.sum() < 5:
                    continue
                    
                x = df_clean.loc[mask, feature]
                y = df_clean.loc[mask, target_col]
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(x, y)
                
                # Spearman correlation  
                spearman_r, spearman_p = spearmanr(x, y)
                
                correlations.append({
                    'feature': feature,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'abs_pearson': abs(pearson_r),
                    'n_samples': mask.sum()
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)
        
        print(f"\nTop 10 Features by Correlation with {target_col}:")
        print("-" * 80)
        for _, row in corr_df.head(10).iterrows():
            significance = "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
            print(f"{row['feature']:<25} | Pearson: {row['pearson_r']:.3f}{significance} | Spearman: {row['spearman_r']:.3f} | n={row['n_samples']}")
        
        print("\n** p < 0.01, * p < 0.05")
        
        return corr_df
    
    def create_feature_target_plots(self, df, corr_df, target_col='energy', output_dir=None):
        """Create scatter plots of key features vs target"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Get top 6 features for plotting
        top_features = corr_df.head(6)['feature'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        df_clean = df.dropna(subset=[target_col])
        
        for i, feature in enumerate(top_features):
            if i >= 6:
                break
                
            # Get clean data for this feature
            mask = df_clean[feature].notna() & df_clean[target_col].notna()
            x = df_clean.loc[mask, feature]
            y = df_clean.loc[mask, target_col]
            
            if len(x) < 2:
                continue
            
            # Scatter plot
            axes[i].scatter(x, y, alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)
            
            # Get correlation for this feature
            corr_row = corr_df[corr_df['feature'] == feature].iloc[0]
            r_val = corr_row['pearson_r']
            
            axes[i].set_xlabel(feature.replace('_', ' ').title())
            axes[i].set_ylabel(target_col.replace('_', ' ').title())
            axes[i].set_title(f'{feature}\nr = {r_val:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_target_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature-target relationship plots saved")
    
    def analyze_feature_distributions(self, df, output_dir=None):
        """Analyze and visualize feature distributions"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Select key structural features
        key_features = [
            'mean_bond_length', 'mean_coordination', 'radius_of_gyration', 
            'asphericity', 'surface_fraction', 'anisotropy'
        ]
        
        # Filter available features
        available_features = [f for f in key_features if f in df.columns]
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:6]):
            data = df[feature].dropna()
            
            if len(data) > 1:
                # Histogram with KDE
                axes[i].hist(data, bins=20, alpha=0.7, density=True, edgecolor='black')
                
                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                median_val = data.median()
                
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                axes[i].axvline(median_val, color='blue', linestyle=':', label=f'Median: {median_val:.3f}')
                
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'{feature}\nσ = {std_val:.3f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Print distribution stats
                print(f"{feature:<20} | Mean: {mean_val:.3f} | Std: {std_val:.3f} | Range: [{data.min():.3f}, {data.max():.3f}]")
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature distribution plots saved")
    
    def perform_pca_analysis(self, df, output_dir=None):
        """Perform PCA analysis and visualization"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)
        
        # Select numeric features (excluding identifiers and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['energy', 'energy_per_atom']]
        
        # Prepare data
        df_pca = df[feature_cols + ['energy']].dropna()
        
        if len(df_pca) < 5:
            print("Not enough samples for PCA analysis")
            return None
        
        X = df_pca[feature_cols]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Print explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"Explained Variance by Component:")
        for i in range(min(5, len(explained_var))):
            print(f"  PC{i+1}: {explained_var[i]:.3f} ({explained_var[i]*100:.1f}%)")
        print(f"  Cumulative (first 3): {cumulative_var[2]:.3f} ({cumulative_var[2]*100:.1f}%)")
        
        # Create PCA plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Explained variance plot
        axes[0,0].bar(range(1, min(11, len(explained_var)+1)), explained_var[:10])
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].set_title('PCA Explained Variance')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Cumulative explained variance
        axes[0,1].plot(range(1, len(cumulative_var)+1), cumulative_var, 'bo-')
        axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='80%')
        axes[0,1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
        axes[0,1].set_xlabel('Number of Components')
        axes[0,1].set_ylabel('Cumulative Explained Variance')
        axes[0,1].set_title('Cumulative Explained Variance')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. PC1 vs PC2 colored by energy
        if 'energy' in df_pca.columns and not df_pca['energy'].isna().all():
            scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                      c=df_pca['energy'], cmap='viridis', alpha=0.6)
            axes[1,0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[1,0].set_title('PCA: PC1 vs PC2 (colored by energy)')
            plt.colorbar(scatter, ax=axes[1,0], label='Energy')
        else:
            axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
            axes[1,0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[1,0].set_title('PCA: PC1 vs PC2')
        
        # 4. Feature loadings for PC1 and PC2
        loadings = pca.components_[:2].T
        feature_importance = np.abs(loadings).sum(axis=1)
        top_features_idx = np.argsort(feature_importance)[-8:]  # Top 8 features
        
        for i, idx in enumerate(top_features_idx):
            axes[1,1].arrow(0, 0, loadings[idx, 0], loadings[idx, 1], 
                          head_width=0.02, head_length=0.02, fc='red', ec='red')
            axes[1,1].text(loadings[idx, 0]*1.1, loadings[idx, 1]*1.1, 
                         feature_cols[idx], fontsize=8)
        
        axes[1,1].set_xlim(-1, 1)
        axes[1,1].set_ylim(-1, 1)
        axes[1,1].set_xlabel('PC1 Loading')
        axes[1,1].set_ylabel('PC2 Loading')
        axes[1,1].set_title('Feature Loadings (PC1 vs PC2)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PCA analysis plots saved")
        
        return pca, X_pca, scaler
    
    def detect_outliers(self, df, output_dir=None):
        """Detect and analyze outliers in the dataset"""
        if output_dir is None:
            output_dir = Path('./plots')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("OUTLIER & VARIABILITY ANALYSIS")
        print("="*60)
        
        # Select key features for outlier analysis
        key_features = ['mean_bond_length', 'mean_coordination', 'radius_of_gyration', 
                       'asphericity', 'surface_fraction', 'anisotropy']
        
        available_features = [f for f in key_features if f in df.columns]
        
        outliers_info = []
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:6]):
            data = df[feature].dropna()
            
            if len(data) > 4:
                # Calculate IQR-based outliers
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_indices = df[df[feature].isin(outliers)].index
                
                # Store outlier info
                if len(outliers) > 0:
                    outliers_info.append({
                        'feature': feature,
                        'n_outliers': len(outliers),
                        'outlier_files': df.loc[outlier_indices, 'filename'].tolist()
                    })
                
                # Box plot
                axes[i].boxplot(data, labels=[feature.replace('_', '\n')])
                axes[i].set_title(f'{feature}\n{len(outliers)} outliers')
                axes[i].grid(True, alpha=0.3)
                
                # Highlight outliers
                if len(outliers) > 0:
                    axes[i].scatter([1]*len(outliers), outliers, color='red', s=50, alpha=0.7, label='Outliers')
                    axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print outlier summary
        if outliers_info:
            print("\nOutlier Summary:")
            print("-" * 50)
            for info in outliers_info:
                print(f"{info['feature']:<20} | {info['n_outliers']} outliers")
                if info['n_outliers'] <= 5:  # Show filenames for small number of outliers
                    for filename in info['outlier_files']:
                        print(f"  → {filename}")
        else:
            print("No significant outliers detected using IQR method")
        
        print(f"Outlier analysis plots saved")
        
        return outliers_info

def main():
    """Main execution function"""
    # Example usage - set default to your data path
    print("Default path: /Users/wilbert/Documents/GitHub/AIAC/data/Au20_OPT_1000")
    data_dir = input("Enter path to xyz files directory (press Enter for default): ").strip()
    if not data_dir:
        data_dir = "/Users/wilbert/Documents/GitHub/AIAC/data/Au20_OPT_1000"  # Your data path
    
    print(f"Using directory: {data_dir}")
    
    # Initialize analyzer
    analyzer = AuClusterAnalyzer(data_dir)
    
    # Parse files
    analyzer.parse_all_files()
    
    if not analyzer.structures:
        print("No structures were successfully parsed!")
        print("Please check:")
        print("1. Directory path is correct")
        print("2. Directory contains .xyz files")
        print("3. .xyz files are properly formatted")
        return None  # Return None explicitly
    
    # Compute descriptors
    df = analyzer.compute_descriptors()
    
    # Generate basic statistics
    summary_stats = analyzer.generate_statistics(df)
    
    # Create output directories
    output_dir = Path('./au_cluster_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save basic results
    analyzer.save_results(df, output_dir)
    
    # Create basic plots
    analyzer.create_plots(df, output_dir)
    
    # Run advanced analysis
    advanced_results = analyzer.run_advanced_analysis(df, output_dir / 'advanced')
    
    print(f"\n🎉 Complete analysis finished!")
    print(f"📁 Basic results: {output_dir}")
    print(f"🔬 Advanced analysis: {output_dir / 'advanced'}")
    
    return df, analyzer, advanced_results

if __name__ == "__main__":
    result = main()
    if result is not None:
        df, analyzer, advanced_results = result
        print("Analysis completed successfully!")
        
        # Print key findings
        if advanced_results['correlations'] is not None:
            top_corr = advanced_results['correlations'].head(3)
            print(f"\n🔍 Key Findings:")
            print(f"Top 3 features correlated with energy:")
            for _, row in top_corr.iterrows():
                print(f"  • {row['feature']}: r = {row['pearson_r']:.3f}")
    else:
        print("Analysis failed - no data to process")