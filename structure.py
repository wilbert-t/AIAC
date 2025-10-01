#!/usr/bin/env python3
"""
Intelligent Structure Selection for ML Training
Finds neat, symmetric, and energetically favorable structures for better model training
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StructureAnalyzer:
    """Analyzes molecular structures for aesthetic and energetic properties"""
    
    def __init__(self, data_dir="data/Au20_OPT_1000"):
        self.data_dir = Path(data_dir)
        self.structures = {}
        self.properties = {}
        
    def load_all_structures(self):
        """Load all available structures with their energies"""
        print("Loading all available structures...")
        
        xyz_files = list(self.data_dir.glob("*.xyz"))
        print(f"Found {len(xyz_files)} structure files")
        
        for xyz_file in xyz_files:
            struct_id = f"structure_{xyz_file.stem}"
            try:
                coords, energy = self._load_xyz_file(xyz_file)
                self.structures[struct_id] = {
                    'coordinates': coords,
                    'energy': energy,
                    'file_path': xyz_file
                }
            except Exception as e:
                print(f"Error loading {xyz_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.structures)} structures")
        return len(self.structures)
    
    def _load_xyz_file(self, xyz_file):
        """Load coordinates and energy from XYZ file"""
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            
        n_atoms = int(lines[0].strip())
        energy = float(lines[1].strip())
        
        coordinates = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append([x, y, z])
        
        return np.array(coordinates), energy
    
    def calculate_structure_properties(self, coordinates):
        """Calculate comprehensive structural properties"""
        n_atoms = len(coordinates)
        center = np.mean(coordinates, axis=0)
        centered_coords = coordinates - center
        
        # Basic geometric properties
        distances_from_center = np.linalg.norm(centered_coords, axis=1)
        radius = np.max(distances_from_center)
        avg_distance_from_center = np.mean(distances_from_center)
        std_distance_from_center = np.std(distances_from_center)
        
        # Pairwise distances and bonds
        pairwise_distances = []
        bonds = []
        bond_lengths = []
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                pairwise_distances.append(dist)
                
                # Au-Au bond threshold
                if 2.3 <= dist <= 3.2:
                    bonds.append((i, j))
                    bond_lengths.append(dist)
        
        # Coordination numbers
        coordination = [0] * n_atoms
        for i, j in bonds:
            coordination[i] += 1
            coordination[j] += 1
        
        # Symmetry analysis - moment of inertia tensor
        inertia_tensor = np.zeros((3, 3))
        for coord in centered_coords:
            inertia_tensor += np.outer(coord, coord)
        
        eigenvalues = np.sort(np.linalg.eigvals(inertia_tensor))
        
        # Shape descriptors
        if eigenvalues[2] > 1e-10:
            compactness = eigenvalues[0] / eigenvalues[2]  # Sphericity measure
            prolate = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
            oblate = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
            anisotropy = (eigenvalues[2] - eigenvalues[0]) / eigenvalues[2]
        else:
            compactness = prolate = oblate = anisotropy = 0
        
        # Symmetry detection
        symmetry_score = self._calculate_symmetry_score(centered_coords)
        
        # Robustness score for perturbation handling (NEW for Task 3)
        robustness_score = self._calculate_robustness_score({
            'avg_coordination': np.mean(coordination),
            'n_bonds': len(bonds),
            'connectivity_density': len(bonds) / n_atoms,
            'coordination_diversity': len(set(coordination)) / n_atoms,
            'energy_per_bond': 0,  # Will be set later with energy
            'structural_flexibility': 1.0 - symmetry_score,  # Lower symmetry = more flexible
            'compactness': compactness
        })
        
        # Neatness score (combination of multiple factors)
        neatness_score = self._calculate_neatness_score({
            'compactness': compactness,
            'std_distance_from_center': std_distance_from_center,
            'radius': radius,
            'bond_uniformity': np.std(bond_lengths) if bond_lengths else 0,
            'coordination_uniformity': np.std(coordination),
            'symmetry_score': symmetry_score
        })
        
        return {
            'n_atoms': n_atoms,
            'radius': radius,
            'avg_distance_from_center': avg_distance_from_center,
            'std_distance_from_center': std_distance_from_center,
            'n_bonds': len(bonds),
            'avg_bond_length': np.mean(bond_lengths) if bond_lengths else 0,
            'std_bond_length': np.std(bond_lengths) if bond_lengths else 0,
            'avg_coordination': np.mean(coordination),
            'std_coordination': np.std(coordination),
            'max_coordination': np.max(coordination),
            'min_coordination': np.min(coordination),
            'compactness': compactness,
            'prolate': prolate,
            'oblate': oblate,
            'anisotropy': anisotropy,
            'symmetry_score': symmetry_score,
            'neatness_score': neatness_score,
            'robustness_score': robustness_score,
            'bonds': bonds,
            'coordination': coordination
        }
    
    def _calculate_symmetry_score(self, centered_coords):
        """Calculate symmetry score based on multiple symmetry operations"""
        n_atoms = len(centered_coords)
        tolerance = 0.2
        
        # Inversion symmetry
        inverted_coords = -centered_coords
        inversion_matches = 0
        for coord in centered_coords:
            min_dist = np.min(np.linalg.norm(inverted_coords - coord, axis=1))
            if min_dist < tolerance:
                inversion_matches += 1
        inversion_symmetry = inversion_matches / n_atoms
        
        # Rotational symmetry (approximate)
        rotation_scores = []
        for angle in [np.pi/2, np.pi/3, np.pi/4, np.pi/6]:
            # Rotation around z-axis
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            rotated_coords = centered_coords @ rotation_matrix.T
            
            matches = 0
            for coord in centered_coords:
                min_dist = np.min(np.linalg.norm(rotated_coords - coord, axis=1))
                if min_dist < tolerance:
                    matches += 1
            rotation_scores.append(matches / n_atoms)
        
        max_rotation_symmetry = np.max(rotation_scores)
        
        # Combined symmetry score
        symmetry_score = 0.5 * inversion_symmetry + 0.5 * max_rotation_symmetry
        return symmetry_score
    
    def _calculate_neatness_score(self, props):
        """Calculate overall neatness score"""
        # Normalize and combine different factors
        # Higher compactness = better
        compactness_score = props['compactness']
        
        # Lower standard deviations = more uniform = better
        uniformity_score = 1.0 / (1.0 + props['std_distance_from_center'] + 
                                 props['bond_uniformity'] + 
                                 props['coordination_uniformity'])
        
        # Higher symmetry = better
        symmetry_score = props['symmetry_score']
        
        # Combine scores (weights can be adjusted)
        neatness_score = (0.4 * compactness_score + 
                         0.3 * uniformity_score + 
                         0.3 * symmetry_score)
        
        return neatness_score
    
    def _calculate_robustness_score(self, props):
        """Calculate structural robustness score for perturbation handling"""
        # Factors that contribute to robustness against atomic perturbations
        
        # Higher coordination = more stable under perturbations
        coordination_score = min(props['avg_coordination'] / 8.0, 1.0)  # Normalize to [0,1]
        
        # Higher connectivity density = more constraints = more stable
        connectivity_score = min(props['connectivity_density'] / 4.0, 1.0)  # Normalize
        
        # More diverse coordination environments = better perturbation tolerance
        diversity_score = props['coordination_diversity']
        
        # Structural flexibility (inverse of symmetry) = better adaptation to changes
        flexibility_score = props['structural_flexibility']
        
        # Compactness helps with stability
        compactness_score = props['compactness']
        
        # Combine robustness factors (weighted for perturbation resistance)
        robustness_score = (0.3 * coordination_score +      # High coordination = stable
                           0.2 * connectivity_score +       # More bonds = more constraints
                           0.2 * diversity_score +          # Diverse environments = adaptable
                           0.2 * flexibility_score +        # Low symmetry = flexible
                           0.1 * compactness_score)         # Compact = stable
        
        return robustness_score
    
    def _calculate_robustness_score_with_energy(self, props):
        """Calculate enhanced robustness score including energy stability"""
        # Higher coordination = more stable under perturbations
        coordination_score = min(props['avg_coordination'] / 8.0, 1.0)
        
        # Higher connectivity density = more constraints = more stable
        connectivity_score = min(props['connectivity_density'] / 4.0, 1.0)
        
        # More diverse coordination environments = better perturbation tolerance
        diversity_score = props['coordination_diversity']
        
        # Structural flexibility (inverse of symmetry) = better adaptation to changes
        flexibility_score = props['structural_flexibility']
        
        # Compactness helps with stability
        compactness_score = props['compactness']
        
        # Energy stability - higher energy per bond = more stable
        energy_stability_score = props['energy_stability']
        
        # Enhanced robustness score for Task 3 (perturbation resistance)
        robustness_score = (0.25 * coordination_score +      # High coordination = stable
                           0.20 * connectivity_score +       # More bonds = more constraints  
                           0.15 * diversity_score +          # Diverse environments = adaptable
                           0.15 * flexibility_score +        # Low symmetry = flexible
                           0.15 * energy_stability_score +   # Strong bonds = stable
                           0.10 * compactness_score)         # Compact = stable
        
        return robustness_score
    
    def analyze_all_structures(self):
        """Analyze all loaded structures"""
        print("Analyzing structural properties...")
        
        for struct_id, struct_data in self.structures.items():
            coords = struct_data['coordinates']
            energy = struct_data['energy']
            
            props = self.calculate_structure_properties(coords)
            props['energy'] = energy
            
            # Update robustness score with energy information
            if props['n_bonds'] > 0:
                energy_per_bond = abs(energy) / props['n_bonds']
                # Higher energy per bond = more stable bonds = more robust
                energy_stability = min(energy_per_bond / 30.0, 1.0)  # Normalize
                
                # Recalculate robustness with energy component
                robustness_factors = {
                    'avg_coordination': props['avg_coordination'],
                    'connectivity_density': props['n_bonds'] / props['n_atoms'],
                    'coordination_diversity': len(set(props['coordination'])) / props['n_atoms'],
                    'structural_flexibility': 1.0 - props['symmetry_score'],
                    'compactness': props['compactness'],
                    'energy_stability': energy_stability
                }
                props['robustness_score'] = self._calculate_robustness_score_with_energy(robustness_factors)
            
            self.properties[struct_id] = props
        
        print(f"Analysis complete for {len(self.properties)} structures")
        return self.properties
    
    def select_high_quality_structures(self, 
                                     n_structures=100,
                                     energy_weight=0.0,
                                     neatness_weight=0.3,
                                     robustness_weight=0.7):
        """Select high-quality structures prioritizing robustness for Task 3"""
        print(f"Selecting top {n_structures} structures with optimal Task 3 weighting...")
        print(f"Weights - Energy: {energy_weight:.0%}, Beauty: {neatness_weight:.0%}, Robustness: {robustness_weight:.0%}")
        
        # Create composite score with robustness focus
        energies = [props['energy'] for props in self.properties.values()]
        neatness_scores = [props['neatness_score'] for props in self.properties.values()]
        robustness_scores = [props['robustness_score'] for props in self.properties.values()]
        
        # Normalize scores (lower energy is better, higher scores are better for others)
        if energy_weight > 0:
            min_energy = np.min(energies)
            max_energy = np.max(energies)
            energy_scores = [(max_energy - energy) / (max_energy - min_energy) 
                            for energy in energies]
        else:
            energy_scores = [0] * len(energies)  # Ignore energy if weight is 0
        
        # Combine scores with robustness focus
        composite_scores = []
        struct_ids = list(self.properties.keys())
        
        for i, struct_id in enumerate(struct_ids):
            composite_score = (energy_weight * energy_scores[i] + 
                             neatness_weight * neatness_scores[i] +
                             robustness_weight * robustness_scores[i])
            composite_scores.append((struct_id, composite_score, 
                                   self.properties[struct_id]))
        
        # Sort by composite score (highest first)
        composite_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top structures
        selected_structures = composite_scores[:n_structures]
        
        print(f"Selected {len(selected_structures)} structures with optimal Task 3 weighting")
        
        # Show robustness statistics
        avg_robustness = np.mean([props['robustness_score'] for _, _, props in selected_structures])
        avg_neatness = np.mean([props['neatness_score'] for _, _, props in selected_structures])
        print(f"Average robustness score: {avg_robustness:.3f}")
        print(f"Average neatness score: {avg_neatness:.3f}")
        
        return selected_structures
    
    def create_improved_dataset(self, selected_structures, output_file):
        """Create improved training dataset CSV"""
        print(f"Creating improved dataset: {output_file}")
        
        # Create DataFrame with selected structures
        data_rows = []
        
        for struct_id, score, props in selected_structures:
            struct_data = self.structures[struct_id]
            coords = struct_data['coordinates']
            
            # Create row with flattened coordinates and robustness info
            row = {
                'structure_id': struct_id,
                'energy': props['energy'],
                'n_atoms': props['n_atoms'],
                'cluster_type': 'Au20',
                'neatness_score': props['neatness_score'],
                'compactness': props['compactness'],
                'symmetry_score': props['symmetry_score'],
                'robustness_score': props['robustness_score'],  # NEW: robustness for Task 3
                'n_bonds': props['n_bonds'],
                'avg_coordination': props['avg_coordination'],
                'max_coordination': props['max_coordination'],
                'min_coordination': props['min_coordination']
            }
            
            # Add flattened coordinates
            for i, (x, y, z) in enumerate(coords, 1):
                row[f'atom_{i}_element'] = 'Au'
                row[f'atom_{i}_x'] = x
                row[f'atom_{i}_y'] = y
                row[f'atom_{i}_z'] = z
            
            data_rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data_rows)
        df.to_csv(output_file, index=False)
        
        print(f"Saved improved dataset with {len(df)} structures to {output_file}")
        return df
    
    def visualize_selection_results(self, selected_structures):
        """Visualize the selection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        energies = [props['energy'] for _, _, props in selected_structures]
        neatness_scores = [props['neatness_score'] for _, _, props in selected_structures]
        compactness = [props['compactness'] for _, _, props in selected_structures]
        symmetry_scores = [props['symmetry_score'] for _, _, props in selected_structures]
        
        # Plot 1: Energy vs Neatness
        axes[0,0].scatter(energies, neatness_scores, alpha=0.6, c='blue')
        axes[0,0].set_xlabel('Energy (eV)')
        axes[0,0].set_ylabel('Neatness Score')
        axes[0,0].set_title('Energy vs Neatness Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Highlight Structure 350 if present
        for struct_id, _, props in selected_structures:
            if struct_id == 'structure_350':
                axes[0,0].scatter([props['energy']], [props['neatness_score']], 
                                c='red', s=100, marker='*', label='Structure 350')
                axes[0,0].legend()
        
        # Plot 2: Compactness vs Symmetry
        axes[0,1].scatter(compactness, symmetry_scores, alpha=0.6, c='green')
        axes[0,1].set_xlabel('Compactness')
        axes[0,1].set_ylabel('Symmetry Score')
        axes[0,1].set_title('Compactness vs Symmetry Score')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Energy distribution
        axes[1,0].hist(energies, bins=20, alpha=0.7, color='orange')
        axes[1,0].set_xlabel('Energy (eV)')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Energy Distribution of Selected Structures')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Neatness distribution
        axes[1,1].hist(neatness_scores, bins=20, alpha=0.7, color='purple')
        axes[1,1].set_xlabel('Neatness Score')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Neatness Score Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_structure_selection.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as: improved_structure_selection.png")
        plt.show()

def main():
    """Main function with optimal Task 3 weighting"""
    print("="*80)
    print("🎯 OPTIMAL TASK 3 STRUCTURE SELECTION")
    print("⚖️  Perfect Balance: 25% Energy + 60% Robustness + 15% Beauty")
    print("="*80)
    
    # Initialize analyzer
    analyzer = StructureAnalyzer()
    
    # Load all structures
    n_loaded = analyzer.load_all_structures()
    if n_loaded == 0:
        print("No structures loaded. Check data directory.")
        return
    
    # Analyze properties including robustness
    analyzer.analyze_all_structures()
    
    # Optimal Task 3 weighting: 25% energy, 60% robustness, 15% beauty
    selection_configs = [
        {'n_structures': 50, 'name': 'elite', 'energy_weight': 0.25, 'neatness_weight': 0.15, 'robustness_weight': 0.60},
        {'n_structures': 100, 'name': 'high_quality', 'energy_weight': 0.25, 'neatness_weight': 0.15, 'robustness_weight': 0.60},
        {'n_structures': 200, 'name': 'balanced', 'energy_weight': 0.25, 'neatness_weight': 0.15, 'robustness_weight': 0.60},
    ]
    
    results = {}
    
    for config in selection_configs:
        print(f"\n--- Selecting {config['name']} structures ---")
        
        selected = analyzer.select_high_quality_structures(
            n_structures=config['n_structures'],
            energy_weight=config['energy_weight'],
            neatness_weight=config['neatness_weight'],
            robustness_weight=config['robustness_weight']
        )
        
        # Check if Structure 350 is included
        struct_350_included = any(sid == 'structure_350' for sid, _, _ in selected)
        print(f"Structure 350 included: {'✅ YES' if struct_350_included else '❌ NO'}")
        
        if struct_350_included:
            struct_350_rank = next(i for i, (sid, _, _) in enumerate(selected, 1) 
                                 if sid == 'structure_350')
            print(f"Structure 350 rank: {struct_350_rank}")
        
        # Create dataset
        output_file = f'improved_dataset_{config["name"]}.csv'
        df = analyzer.create_improved_dataset(selected, output_file)
        
        results[config['name']] = {
            'selected_structures': selected,
            'dataframe': df,
            'struct_350_included': struct_350_included
        }
        
        # Show top 10 with robustness info
        print(f"\nTop 10 {config['name']} structures:")
        for i, (struct_id, score, props) in enumerate(selected[:10], 1):
            print(f"  {i:2d}. {struct_id:<15} Score: {score:.3f} "
                  f"Energy: {props['energy']:.3f} eV Robustness: {props['robustness_score']:.3f} "
                  f"Beauty: {props['neatness_score']:.3f}")
    
    # Visualize best selection
    print(f"\nVisualizing 'elite' selection results...")
    analyzer.visualize_selection_results(results['elite']['selected_structures'])
    
    print(f"\n" + "="*80)
    print("✅ OPTIMAL TASK 3 DATASETS CREATED")
    print("="*80)
    print("Files created with perfect 25/60/15 weighting:")
    for config in selection_configs:
        filename = f"improved_dataset_{config['name']}.csv"
        struct_350 = "✅" if results[config['name']]['struct_350_included'] else "❌"
        avg_robustness = np.mean([props['robustness_score'] for _, _, props in results[config['name']]['selected_structures']])
        print(f"  - {filename:<35} (Structure 350: {struct_350}) Avg Robustness: {avg_robustness:.3f}")
    
    print(f"\n🎯 TASK 3 RECOMMENDATION:")
    print(f"Use 'improved_dataset_elite.csv' for perturbation studies!")
    print(f"This dataset uses OPTIMAL weighting: 25% Energy + 60% Robustness + 15% Beauty")

if __name__ == "__main__":
    main()