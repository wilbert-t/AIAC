#!/usr/bin/env python3
"""
Task 1: Au Cluster Analysis - IMPROVED VERSION
Enhanced descriptors with better surface detection, multivariate outliers, and SOAP integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
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

# SOAP descriptors
try:
    from dscribe.descriptors import SOAP
    SOAP_AVAILABLE = True
except ImportError:
    print("Warning: DScribe not available. SOAP descriptors will be skipped.")
    SOAP_AVAILABLE = False

class ImprovedAuClusterAnalyzer:
    
    def __init__(self, data_dir, surface_threshold_mode='bulk_reference'):
        self.data_dir = Path(data_dir)
        self.structures = []
        self.surface_threshold_mode = surface_threshold_mode
        self.soap_features = None
        
        # Bulk Au coordination reference (fcc = 12)
        self.bulk_coordination_au = 12
        
    def parse_xyz_file(self, filepath):
        """Parse single xyz file using ASE - robust and accurate"""
        try:
            atoms = read(str(filepath))
            
            # Extract energy from comment line
            energy = atoms.info.get('energy', None)
            
            if energy is None:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 1:
                    energy = self._extract_energy(lines[1])
            
            return {
                'filename': filepath.name,
                'n_atoms': len(atoms),
                'energy': energy,
                'atoms': atoms,
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
                if -50000 < val < 50000:  # Au cluster energy range
                    return val
            except:
                continue
        return None
    
    def parse_all_files(self):
        """Parse all xyz files in directory"""
        if not self.data_dir.exists():
            print(f"Directory does not exist: {self.data_dir}")
            return []
            
        xyz_files = list(self.data_dir.glob("*.xyz"))
        if len(xyz_files) == 0:
            xyz_files = list(self.data_dir.glob("**/*.xyz"))
        
        print(f"Found {len(xyz_files)} xyz files")
        
        for filepath in xyz_files:
            structure = self.parse_xyz_file(filepath)
            if structure is not None:
                self.structures.append(structure)
        
        print(f"Successfully parsed {len(self.structures)} files")
        return self.structures
    
    def compute_bond_lengths_ase(self, atoms):
        """Compute bond lengths using ASE neighbor list"""
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
        cutoffs = natural_cutoffs(atoms, mult=1.1)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        
        coord_nums = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            coord_nums.append(len(indices))
        
        return np.array(coord_nums)
    
    def compute_surface_atoms_improved(self, atoms, threshold_mode='bulk_reference'):
        """
        IMPROVED surface atom detection
        
        Modes:
        - 'bulk_reference': Use bulk Au coordination (12) as reference
        - 'adaptive': Use mean-based threshold (original method)
        - 'size_scaled': Scale threshold based on cluster size
        """
        coord_nums = self.compute_coordination_numbers_ase(atoms)
        
        if threshold_mode == 'bulk_reference':
            # Surface atoms have coordination < 75% of bulk (≤ 9 for Au)
            surface_threshold = 0.75 * self.bulk_coordination_au  # 9
            
        elif threshold_mode == 'size_scaled':
            # Scale threshold based on cluster size
            n_atoms = len(atoms)
            if n_atoms <= 20:
                surface_threshold = 6  # Small clusters
            elif n_atoms <= 100:
                surface_threshold = 8  # Medium clusters
            else:
                surface_threshold = 9  # Large clusters
                
        else:  # 'adaptive' (original method)
            mean_coord = np.mean(coord_nums)
            surface_threshold = mean_coord - 1.0
        
        surface_atoms = np.sum(coord_nums <= surface_threshold)
        
        return {
            'surface_fraction': surface_atoms / len(atoms),
            'surface_threshold': surface_threshold,
            'surface_count': surface_atoms
        }
    
    def compute_gyration_tensor_properties(self, atoms):
        """
        IMPROVED geometric properties using gyration tensor
        More robust than bounding box ranges
        """
        positions = atoms.get_positions()
        com = atoms.get_center_of_mass()
        
        # Center positions
        centered = positions - com
        
        # Gyration tensor
        gyration_tensor = np.dot(centered.T, centered) / len(atoms)
        eigenvals = np.linalg.eigvals(gyration_tensor)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        
        # Robust geometric descriptors
        rg_squared = np.sum(eigenvals)
        radius_of_gyration = np.sqrt(rg_squared)
        
        # Improved anisotropy (outlier-resistant)
        if eigenvals[2] > 1e-8:  # Avoid division by zero
            anisotropy_tensor = eigenvals[0] / eigenvals[2]
        else:
            anisotropy_tensor = eigenvals[0] / (eigenvals[2] + 1e-8)
        
        # Asphericity
        asphericity = eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])
        if rg_squared > 0:
            asphericity_normalized = asphericity / rg_squared
        else:
            asphericity_normalized = 0.0
        
        # Triaxial ratios
        c_over_a = np.sqrt(eigenvals[2] / eigenvals[0]) if eigenvals[0] > 0 else 0
        b_over_a = np.sqrt(eigenvals[1] / eigenvals[0]) if eigenvals[0] > 0 else 0
        
        return {
            'radius_of_gyration': radius_of_gyration,
            'asphericity': asphericity_normalized,
            'anisotropy_tensor': anisotropy_tensor,  # Improved version
            'triaxial_c_a': c_over_a,
            'triaxial_b_a': b_over_a,
            'gyration_eigenvals': eigenvals
        }
    
    def compute_radial_distribution_function(self, atoms, r_max=10.0, n_bins=20):
        """
        Compute radial distribution function (RDF) features
        Provides richer local environment information
        """
        positions = atoms.get_positions()
        n_atoms = len(positions)
        
        # Compute all pairwise distances
        distances = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= r_max:
                    distances.append(dist)
        
        if len(distances) == 0:
            return {'rdf_features': np.zeros(n_bins)}
        
        # Create histogram (RDF)
        hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
        
        # Normalize by number of pairs and bin volume
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Volume normalization for spherical shells
        for i in range(n_bins):
            r = bin_centers[i]
            if r > 0:
                shell_volume = 4 * np.pi * r**2 * bin_width
                hist[i] = hist[i] / shell_volume
        
        rdf_features = {}
        for i, val in enumerate(hist):
            rdf_features[f'rdf_bin_{i+1}'] = val
        
        return rdf_features
    
    def compute_sphericity_index(self, atoms):
        """
        Compute sphericity index = surface area of equivalent sphere / actual surface area
        More accurate than simple compactness
        """
        positions = atoms.get_positions()
        
        # Estimate surface area using convex hull (approximation)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            surface_area_approx = hull.area
        except:
            # Fallback: use bounding sphere approximation
            ranges = np.max(positions, axis=0) - np.min(positions, axis=0)
            max_range = np.max(ranges)
            surface_area_approx = 4 * np.pi * (max_range/2)**2
        
        # Volume-equivalent sphere
        n_atoms = len(atoms)
        atom_volume = 4/3 * np.pi * (1.44)**3  # Au atomic radius ≈ 1.44 Å
        total_volume = n_atoms * atom_volume
        
        # Radius of equivalent sphere
        equiv_radius = (3 * total_volume / (4 * np.pi))**(1/3)
        equiv_surface_area = 4 * np.pi * equiv_radius**2
        
        # Sphericity index
        if surface_area_approx > 0:
            sphericity = equiv_surface_area / surface_area_approx
        else:
            sphericity = 1.0
        
        return {
            'sphericity_index': sphericity,
            'equiv_radius': equiv_radius,
            'surface_area_approx': surface_area_approx
        }
    
    def create_soap_features(self):
        """
        Create SOAP descriptors with PCA dimensionality reduction
        Retains 95% variance as recommended
        """
        if not SOAP_AVAILABLE:
            print("SOAP descriptors skipped - DScribe not available")
            return None
        
        print("Creating SOAP descriptors...")
        
        # SOAP parameters for Au clusters
        soap = SOAP(
            species=['Au'],
            r_cut=6.0,      # Au interaction range
            n_max=8,        # Radial basis
            l_max=6,        # Angular basis
            sigma=0.5,      # Gaussian width
            periodic=False, # Clusters
            sparse=False,   # Dense for PCA
            average='inner' # Average over atoms
        )
        
        soap_features = []
        filenames = []
        
        for structure in self.structures:
            try:
                atoms = structure['atoms']
                soap_desc = soap.create(atoms)
                soap_features.append(soap_desc)
                filenames.append(structure['filename'])
            except Exception as e:
                print(f"SOAP error for {structure['filename']}: {e}")
                continue
        
        if not soap_features:
            return None
        
        # Convert to array and apply PCA
        soap_array = np.array(soap_features)
        print(f"Original SOAP dimensions: {soap_array.shape}")
        
        # Standardize before PCA
        scaler = StandardScaler()
        soap_scaled = scaler.fit_transform(soap_array)
        
        # PCA to retain 95% variance
        pca = PCA(n_components=0.95, random_state=42)
        soap_pca = pca.fit_transform(soap_scaled)
        
        print(f"SOAP after PCA (95% variance): {soap_pca.shape}")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Create DataFrame
        soap_df = pd.DataFrame(
            soap_pca,
            columns=[f'soap_pc_{i+1}' for i in range(soap_pca.shape[1])]
        )
        soap_df['filename'] = filenames
        
        self.soap_features = [col for col in soap_df.columns if col.startswith('soap_')]
        self.soap_pca_info = {
            'pca': pca,
            'scaler': scaler,
            'n_components': soap_pca.shape[1],
            'explained_variance': pca.explained_variance_ratio_.sum()
        }
        
        return soap_df
    
    def compute_descriptors(self):
        """Compute all enhanced descriptors"""
        descriptors = []
        
        print(f"Computing ENHANCED descriptors for {len(self.structures)} structures...")
        
        # Create SOAP features first
        soap_df = self.create_soap_features()
        
        for structure in self.structures:
            atoms = structure['atoms']
            
            # Basic properties
            desc = {
                'filename': structure['filename'],
                'n_atoms': structure['n_atoms'],
                'energy': structure['energy'],
                'energy_per_atom': structure['energy'] / structure['n_atoms'] if structure['energy'] else None
            }
            
            # Bond statistics
            bonds = self.compute_bond_lengths_ase(atoms)
            if len(bonds) > 0:
                desc.update({
                    'mean_bond_length': np.mean(bonds),
                    'std_bond_length': np.std(bonds),
                    'min_bond_length': np.min(bonds),
                    'max_bond_length': np.max(bonds),
                    'n_bonds': len(bonds),
                    'bond_variance': np.var(bonds)  # Keep for continuity
                })
            else:
                desc.update({
                    'mean_bond_length': None, 'std_bond_length': None,
                    'min_bond_length': None, 'max_bond_length': None,
                    'n_bonds': 0, 'bond_variance': 0.0
                })
            
            # Coordination statistics
            coord_nums = self.compute_coordination_numbers_ase(atoms)
            desc.update({
                'mean_coordination': np.mean(coord_nums),
                'std_coordination': np.std(coord_nums),
                'max_coordination': np.max(coord_nums),
                'min_coordination': np.min(coord_nums)
            })
            
            # IMPROVED surface detection
            surface_info = self.compute_surface_atoms_improved(atoms, self.surface_threshold_mode)
            desc.update(surface_info)
            
            # IMPROVED geometric properties (gyration tensor)
            geom_props = self.compute_gyration_tensor_properties(atoms)
            desc.update(geom_props)
            
            # RDF features
            rdf_features = self.compute_radial_distribution_function(atoms)
            desc.update(rdf_features)
            
            # Sphericity index
            sphericity_info = self.compute_sphericity_index(atoms)
            desc.update(sphericity_info)
            
            # Legacy descriptors (for compatibility)
            positions = atoms.get_positions()
            ranges = np.max(positions, axis=0) - np.min(positions, axis=0)
            desc.update({
                'x_range': ranges[0],
                'y_range': ranges[1], 
                'z_range': ranges[2],
                'max_range': np.max(ranges),
                'anisotropy': np.max(ranges) / np.min(ranges) if np.min(ranges) > 0 else 1.0,  # Keep legacy
                'volume_estimate': np.prod(ranges),
                'compactness': len(atoms) / (np.prod(ranges) + 1e-8)
            })
            
            descriptors.append(desc)
        
        df = pd.DataFrame(descriptors)
        
        # Merge with SOAP features if available
        if soap_df is not None:
            df = df.merge(soap_df, on='filename', how='left')
            print(f"Added {len(self.soap_features)} SOAP PCA features")
        
        return df
    
    def detect_multivariate_outliers(self, df, method='both'):
        """
        IMPROVED multivariate outlier detection
        Methods: 'mahalanobis', 'isolation_forest', 'both'
        """
        print("\n" + "="*60)
        print("MULTIVARIATE OUTLIER DETECTION")
        print("="*60)
        
        # Select numeric features (excluding identifiers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['energy', 'energy_per_atom']]
        
        # Clean data
        df_clean = df[feature_cols].dropna()
        
        if len(df_clean) < 10:
            print("Not enough samples for multivariate outlier detection")
            return df
        
        # Standardize features (CRITICAL for outlier detection)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
        
        outlier_scores = {}
        
        # 1. Mahalanobis distance
        if method in ['mahalanobis', 'both']:
            try:
                cov = EmpiricalCovariance().fit(X_scaled)
                mahal_dist = cov.mahalanobis(X_scaled)
                
                # Threshold: 97.5th percentile (2.5% outliers)
                threshold = np.percentile(mahal_dist, 97.5)
                mahal_outliers = mahal_dist > threshold
                
                outlier_scores['mahalanobis'] = mahal_dist
                outlier_scores['mahalanobis_outliers'] = mahal_outliers
                
                print(f"Mahalanobis outliers: {np.sum(mahal_outliers)} ({np.sum(mahal_outliers)/len(df_clean)*100:.1f}%)")
                
            except Exception as e:
                print(f"Mahalanobis distance failed: {e}")
        
        # 2. Isolation Forest
        if method in ['isolation_forest', 'both']:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            iso_outliers = iso_forest.fit_predict(X_scaled) == -1
            iso_scores = iso_forest.score_samples(X_scaled)
            
            outlier_scores['isolation_forest'] = -iso_scores  # Convert to distance
            outlier_scores['isolation_outliers'] = iso_outliers
            
            print(f"Isolation Forest outliers: {np.sum(iso_outliers)} ({np.sum(iso_outliers)/len(df_clean)*100:.1f}%)")
        
        # Add outlier information to dataframe
        df_result = df.copy()
        
        for method_name, scores in outlier_scores.items():
            if method_name.endswith('_outliers'):
                continue
            
            # Align with original dataframe
            outlier_col = f'{method_name}_score'
            df_result[outlier_col] = np.nan
            df_result.loc[df_clean.index, outlier_col] = scores
            
            # Add binary outlier column
            outlier_binary_col = f'{method_name}_outlier'
            df_result[outlier_binary_col] = False
            outliers_key = f'{method_name}_outliers'
            if outliers_key in outlier_scores:
                df_result.loc[df_clean.index, outlier_binary_col] = outlier_scores[outliers_key]
        
        return df_result
    
    def analyze_feature_correlations_standardized(self, df, target_col='energy'):
        """
        IMPROVED correlation analysis with standardized features
        Addresses the scaling issue mentioned in recommendations
        """
        if target_col not in df.columns or df[target_col].isna().all():
            print(f"Target column '{target_col}' not available")
            return None
            
        print("\n" + "="*60)
        print("STANDARDIZED FEATURE-TARGET CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in [target_col, 'energy_per_atom']]
        
        # Clean data
        df_clean = df[feature_cols + [target_col]].dropna()
        
        if len(df_clean) < 5:
            print("Not enough samples for correlation analysis")
            return None
        
        # STANDARDIZE FEATURES (z-score normalization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[feature_cols])
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_clean.index)
        
        correlations = []
        
        for i, feature in enumerate(feature_cols):
            if df_clean[feature].notna().sum() > 5:
                # Use standardized features
                x_scaled = X_scaled_df[feature]
                y = df_clean[target_col]
                
                # Remove any remaining NaN values
                mask = x_scaled.notna() & y.notna()
                if mask.sum() < 5:
                    continue
                
                x_clean = x_scaled[mask]
                y_clean = y[mask]
                
                try:
                    # Correlations with standardized features
                    pearson_r, pearson_p = pearsonr(x_clean, y_clean)
                    spearman_r, spearman_p = spearmanr(x_clean, y_clean)
                    
                    correlations.append({
                        'feature': feature,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'abs_pearson': abs(pearson_r),
                        'n_samples': mask.sum()
                    })
                except Exception as e:
                    print(f"Correlation error for {feature}: {e}")
                    continue
        
        if not correlations:
            print("No valid correlations found")
            return None
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)
        
        print(f"\nTop 10 Features by Correlation with {target_col} (STANDARDIZED):")
        print("-" * 80)
        for _, row in corr_df.head(10).iterrows():
            significance = "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
            print(f"{row['feature']:<30} | r = {row['pearson_r']:6.3f}{significance} | ρ = {row['spearman_r']:6.3f} | n={row['n_samples']}")
        
        print("\n** p < 0.01, * p < 0.05")
        
        return corr_df
    
    def create_enhanced_visualizations(self, df, output_dir):
        """Create enhanced visualizations including multivariate outliers"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Multivariate outlier visualization
        self._plot_multivariate_outliers(df, output_dir)
        
        # 2. RDF features visualization
        self._plot_rdf_features(df, output_dir)
        
        # 3. Improved geometric properties comparison
        self._plot_geometric_improvements(df, output_dir)
        
        # 4. SOAP PCA visualization
        if self.soap_features:
            self._plot_soap_analysis(df, output_dir)
        
        print(f"Enhanced visualizations saved to {output_dir}")
    
    def _plot_multivariate_outliers(self, df, output_dir):
        """Plot multivariate outlier analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Check for outlier columns
        outlier_cols = [col for col in df.columns if '_score' in col]
        
        if not outlier_cols:
            axes[0,0].text(0.5, 0.5, 'No outlier scores\navailable', ha='center', va='center')
            axes[0,0].set_title('Multivariate Outlier Detection')
        else:
            # Mahalanobis distances
            if 'mahalanobis_score' in df.columns:
                scores = df['mahalanobis_score'].dropna()
                outliers = df['mahalanobis_outlier'].fillna(False)
                
                axes[0,0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
                if len(scores) > 0:
                    threshold = np.percentile(scores, 97.5)
                    axes[0,0].axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
                axes[0,0].set_xlabel('Mahalanobis Distance')
                axes[0,0].set_ylabel('Frequency')
                axes[0,0].set_title('Mahalanobis Distance Distribution')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
            
            # Isolation Forest scores
            if 'isolation_forest_score' in df.columns:
                scores = df['isolation_forest_score'].dropna()
                
                axes[0,1].hist(scores, bins=20, alpha=0.7, edgecolor='black', color='orange')
                axes[0,1].set_xlabel('Isolation Forest Anomaly Score')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title('Isolation Forest Score Distribution')
                axes[0,1].grid(True, alpha=0.3)
        
        # Outlier comparison by energy
        if 'energy' in df.columns:
            for i, method in enumerate(['mahalanobis', 'isolation_forest']):
                outlier_col = f'{method}_outlier'
                if outlier_col in df.columns:
                    ax = axes[1, i]
                    
                    # Separate outliers and normal points
                    normal_mask = ~df[outlier_col].fillna(False)
                    outlier_mask = df[outlier_col].fillna(False)
                    
                    if normal_mask.sum() > 0:
                        ax.scatter(df.loc[normal_mask, 'n_atoms'], df.loc[normal_mask, 'energy'], 
                                 alpha=0.6, label='Normal', s=50)
                    
                    if outlier_mask.sum() > 0:
                        ax.scatter(df.loc[outlier_mask, 'n_atoms'], df.loc[outlier_mask, 'energy'], 
                                 color='red', alpha=0.8, label='Outliers', s=100, marker='x')
                    
                    ax.set_xlabel('Number of Atoms')
                    ax.set_ylabel('Energy (eV)')
                    ax.set_title(f'{method.title()} Outliers vs Energy')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'multivariate_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rdf_features(self, df, output_dir):
        """Plot RDF features analysis"""
        rdf_cols = [col for col in df.columns if col.startswith('rdf_bin_')]
        
        if not rdf_cols:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Average RDF
        rdf_data = df[rdf_cols].values
        mean_rdf = np.nanmean(rdf_data, axis=0)
        std_rdf = np.nanstd(rdf_data, axis=0)
        
        bin_centers = np.arange(len(rdf_cols)) + 1
        
        axes[0,0].errorbar(bin_centers, mean_rdf, yerr=std_rdf, 
                          capsize=3, marker='o', alpha=0.7)
        axes[0,0].set_xlabel('RDF Bin Number')
        axes[0,0].set_ylabel('RDF Intensity')
        axes[0,0].set_title('Average Radial Distribution Function')
        axes[0,0].grid(True, alpha=0.3)
        
        # RDF variance
        rdf_variance = np.nanvar(rdf_data, axis=0)
        axes[0,1].bar(bin_centers, rdf_variance, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('RDF Bin Number')
        axes[0,1].set_ylabel('RDF Variance')
        axes[0,1].set_title('RDF Feature Variance')
        axes[0,1].grid(True, alpha=0.3)
        
        # Correlation with energy
        if 'energy' in df.columns:
            correlations = []
            for col in rdf_cols:
                data_clean = df[[col, 'energy']].dropna()
                if len(data_clean) > 5:
                    corr, _ = pearsonr(data_clean[col], data_clean['energy'])
                    correlations.append(corr)
                else:
                    correlations.append(0)
            
            axes[1,0].bar(bin_centers, correlations, alpha=0.7, color='green')
            axes[1,0].set_xlabel('RDF Bin Number')
            axes[1,0].set_ylabel('Correlation with Energy')
            axes[1,0].set_title('RDF-Energy Correlations')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # RDF heatmap for clusters by size
        size_groups = df.groupby('n_atoms')
        if len(size_groups) > 1:
            size_rdf_means = []
            size_labels = []
            
            for size, group in size_groups:
                if len(group) >= 3:  # At least 3 samples
                    group_rdf = group[rdf_cols].mean().values
                    size_rdf_means.append(group_rdf)
                    size_labels.append(f'{size} atoms')
            
            if len(size_rdf_means) > 1:
                rdf_matrix = np.array(size_rdf_means)
                im = axes[1,1].imshow(rdf_matrix, cmap='viridis', aspect='auto')
                axes[1,1].set_xlabel('RDF Bin Number')
                axes[1,1].set_ylabel('Cluster Size')
                axes[1,1].set_title('RDF by Cluster Size')
                axes[1,1].set_yticks(range(len(size_labels)))
                axes[1,1].set_yticklabels(size_labels)
                plt.colorbar(im, ax=axes[1,1], label='RDF Intensity')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rdf_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_geometric_improvements(self, df, output_dir):
        """Compare old vs new geometric descriptors"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Anisotropy comparison: old (range-based) vs new (tensor-based)
        if all(col in df.columns for col in ['anisotropy', 'anisotropy_tensor']):
            axes[0,0].scatter(df['anisotropy'], df['anisotropy_tensor'], alpha=0.6)
            axes[0,0].plot([0, df[['anisotropy', 'anisotropy_tensor']].max().max()], 
                          [0, df[['anisotropy', 'anisotropy_tensor']].max().max()], 'r--', alpha=0.5)
            axes[0,0].set_xlabel('Range-based Anisotropy (Old)')
            axes[0,0].set_ylabel('Tensor-based Anisotropy (New)')
            axes[0,0].set_title('Anisotropy Method Comparison')
            axes[0,0].grid(True, alpha=0.3)
        
        # Surface detection improvement
        if 'surface_threshold' in df.columns:
            thresholds = df['surface_threshold'].dropna()
            if len(thresholds) > 0:
                axes[0,1].hist(thresholds, bins=10, alpha=0.7, edgecolor='black')
                axes[0,1].set_xlabel('Surface Detection Threshold')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title('Surface Threshold Distribution')
                axes[0,1].grid(True, alpha=0.3)
        
        # Sphericity vs compactness
        if all(col in df.columns for col in ['sphericity_index', 'compactness']):
            axes[0,2].scatter(df['compactness'], df['sphericity_index'], alpha=0.6)
            axes[0,2].set_xlabel('Compactness (Old)')
            axes[0,2].set_ylabel('Sphericity Index (New)')
            axes[0,2].set_title('Shape Descriptor Comparison')
            axes[0,2].grid(True, alpha=0.3)
        
        # Triaxial ratios
        if all(col in df.columns for col in ['triaxial_c_a', 'triaxial_b_a']):
            scatter = axes[1,0].scatter(df['triaxial_c_a'], df['triaxial_b_a'], 
                                      c=df['energy'] if 'energy' in df.columns else 'blue', 
                                      cmap='viridis', alpha=0.6)
            axes[1,0].set_xlabel('c/a ratio')
            axes[1,0].set_ylabel('b/a ratio')
            axes[1,0].set_title('Triaxial Shape Ratios')
            axes[1,0].grid(True, alpha=0.3)
            if 'energy' in df.columns:
                plt.colorbar(scatter, ax=axes[1,0], label='Energy')
        
        # Surface fraction vs coordination
        if all(col in df.columns for col in ['surface_fraction', 'mean_coordination']):
            axes[1,1].scatter(df['mean_coordination'], df['surface_fraction'], alpha=0.6)
            axes[1,1].set_xlabel('Mean Coordination Number')
            axes[1,1].set_ylabel('Surface Fraction')
            axes[1,1].set_title('Coordination vs Surface')
            axes[1,1].grid(True, alpha=0.3)
        
        # Feature importance comparison
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'energy' in df.columns and len(numeric_cols) > 5:
            correlations = []
            feature_names = []
            
            for col in numeric_cols:
                if col not in ['energy', 'energy_per_atom'] and df[col].notna().sum() > 5:
                    clean_data = df[[col, 'energy']].dropna()
                    if len(clean_data) > 5:
                        corr, _ = pearsonr(clean_data[col], clean_data['energy'])
                        correlations.append(abs(corr))
                        feature_names.append(col)
            
            if correlations:
                # Sort by absolute correlation
                sorted_indices = np.argsort(correlations)[-10:]  # Top 10
                top_corrs = [correlations[i] for i in sorted_indices]
                top_features = [feature_names[i] for i in sorted_indices]
                
                axes[1,2].barh(range(len(top_corrs)), top_corrs, alpha=0.7)
                axes[1,2].set_yticks(range(len(top_features)))
                axes[1,2].set_yticklabels([f.replace('_', ' ') for f in top_features], fontsize=8)
                axes[1,2].set_xlabel('|Correlation with Energy|')
                axes[1,2].set_title('Top Feature Correlations')
                axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'geometric_improvements.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_soap_analysis(self, df, output_dir):
        """Plot SOAP PCA analysis"""
        if not self.soap_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # SOAP PCA explained variance
        if hasattr(self, 'soap_pca_info'):
            pca_info = self.soap_pca_info
            explained_var = pca_info['pca'].explained_variance_ratio_
            
            axes[0,0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7)
            axes[0,0].set_xlabel('SOAP PCA Component')
            axes[0,0].set_ylabel('Explained Variance Ratio')
            axes[0,0].set_title(f'SOAP PCA Components\n(Total: {pca_info["explained_variance"]:.3f} variance)')
            axes[0,0].grid(True, alpha=0.3)
            
            # Cumulative variance
            cumsum = np.cumsum(explained_var)
            axes[0,1].plot(range(1, len(cumsum)+1), cumsum, 'bo-', alpha=0.7)
            axes[0,1].axhline(y=0.95, color='red', linestyle='--', label='95%')
            axes[0,1].set_xlabel('Number of Components')
            axes[0,1].set_ylabel('Cumulative Explained Variance')
            axes[0,1].set_title('SOAP PCA Cumulative Variance')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # SOAP features vs energy
        if 'energy' in df.columns and len(self.soap_features) >= 2:
            pc1 = self.soap_features[0]
            pc2 = self.soap_features[1]
            
            if all(col in df.columns for col in [pc1, pc2]):
                scatter = axes[1,0].scatter(df[pc1], df[pc2], c=df['energy'], 
                                          cmap='viridis', alpha=0.6)
                axes[1,0].set_xlabel(f'{pc1}')
                axes[1,0].set_ylabel(f'{pc2}')
                axes[1,0].set_title('SOAP PC1 vs PC2 (Energy Colored)')
                plt.colorbar(scatter, ax=axes[1,0], label='Energy')
        
        # SOAP feature correlations with energy
        if 'energy' in df.columns:
            correlations = []
            soap_names = []
            
            for soap_col in self.soap_features[:10]:  # Top 10 components
                if soap_col in df.columns:
                    clean_data = df[[soap_col, 'energy']].dropna()
                    if len(clean_data) > 5:
                        corr, _ = pearsonr(clean_data[soap_col], clean_data['energy'])
                        correlations.append(corr)
                        soap_names.append(soap_col.replace('soap_pc_', 'PC'))
            
            if correlations:
                axes[1,1].bar(range(len(correlations)), correlations, alpha=0.7)
                axes[1,1].set_xlabel('SOAP Component')
                axes[1,1].set_ylabel('Correlation with Energy')
                axes[1,1].set_title('SOAP-Energy Correlations')
                axes[1,1].set_xticks(range(len(soap_names)))
                axes[1,1].set_xticklabels(soap_names, rotation=45)
                axes[1,1].grid(True, alpha=0.3)
                axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'soap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_enhanced_results(self, df, output_dir):
        """Save all enhanced results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save main descriptors
        df.to_csv(output_dir / 'enhanced_descriptors.csv', index=False)
        
        # Save SOAP PCA info if available
        if hasattr(self, 'soap_pca_info'):
            import json
            soap_info = {
                'n_components': self.soap_pca_info['n_components'],
                'explained_variance': self.soap_pca_info['explained_variance'],
                'component_names': self.soap_features
            }
            with open(output_dir / 'soap_pca_info.json', 'w') as f:
                json.dump(soap_info, f, indent=2)
        
        # Save improvement summary
        improvements = {
            'surface_detection': f'Using {self.surface_threshold_mode} method',
            'anisotropy': 'Tensor-based (outlier resistant)',
            'sphericity': 'Volume-equivalent sphere ratio',
            'rdf_features': f'{len([c for c in df.columns if c.startswith("rdf_")])} bins',
            'soap_features': f'{len(self.soap_features) if self.soap_features else 0} PCA components',
            'outlier_detection': 'Multivariate (Mahalanobis + Isolation Forest)',
            'feature_scaling': 'Z-score standardization for correlations'
        }
        
        with open(output_dir / 'improvements_summary.json', 'w') as f:
            json.dump(improvements, f, indent=2)
        
        print(f"Enhanced results saved to {output_dir}")
        return improvements

def main():
    """Main execution with improvements"""
    print("🔬 IMPROVED Au Cluster Analysis with Enhanced Descriptors")
    print("="*70)
    print("Key Improvements:")
    print("✅ Bulk reference surface detection (robust for all sizes)")
    print("✅ Gyration tensor anisotropy (outlier-resistant)")
    print("✅ RDF features (richer local environment)")
    print("✅ Sphericity index (better than compactness)")
    print("✅ SOAP descriptors with PCA (95% variance)")
    print("✅ Multivariate outlier detection")
    print("✅ Standardized correlation analysis")
    print("="*70)
    
    # Get data directory
    print("Default path: /Users/wilbert/Documents/GitHub/AIAC/data/Au20_OPT_1000")
    data_dir = input("Enter path to xyz files directory (press Enter for default): ").strip()
    if not data_dir:
        data_dir = "/Users/wilbert/Documents/GitHub/AIAC/data/Au20_OPT_1000"
    
    print(f"Using directory: {data_dir}")
    
    # Initialize improved analyzer
    analyzer = ImprovedAuClusterAnalyzer(data_dir, surface_threshold_mode='bulk_reference')
    
    # Parse files
    analyzer.parse_all_files()
    
    if not analyzer.structures:
        print("No structures were successfully parsed!")
        return None
    
    # Compute enhanced descriptors
    df = analyzer.compute_descriptors()
    
    # Detect multivariate outliers
    df = analyzer.detect_multivariate_outliers(df, method='both')
    
    # Standardized correlation analysis
    corr_df = analyzer.analyze_feature_correlations_standardized(df)
    
    # Create output directory
    output_dir = Path('./improved_au_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save enhanced results
    improvements = analyzer.save_enhanced_results(df, output_dir)
    
    # Create enhanced visualizations
    analyzer.create_enhanced_visualizations(df, output_dir / 'enhanced_plots')
    
    # Save correlation results
    if corr_df is not None:
        corr_df.to_csv(output_dir / 'standardized_correlations.csv', index=False)
    
    print(f"\n🎉 IMPROVED analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Print key improvements achieved
    print(f"\n💡 Improvements Summary:")
    for key, value in improvements.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    if corr_df is not None:
        print(f"\n🔍 Top 3 Standardized Correlations with Energy:")
        for _, row in corr_df.head(3).iterrows():
            print(f"  • {row['feature']}: r = {row['pearson_r']:.3f}")
    
    # Outlier summary
    outlier_methods = ['mahalanobis', 'isolation_forest']
    print(f"\n🚨 Outlier Detection Summary:")
    for method in outlier_methods:
        outlier_col = f'{method}_outlier'
        if outlier_col in df.columns:
            n_outliers = df[outlier_col].sum()
            percentage = n_outliers / len(df) * 100
            print(f"  • {method.title()}: {n_outliers} outliers ({percentage:.1f}%)")
    
    return df, analyzer, corr_df

if __name__ == "__main__":
    result = main()
    if result is not None:
        df, analyzer, corr_df = result
        print("\n✅ Improved analysis completed successfully!")
    else:
        print("\n❌ Analysis failed - no data to process")