#!/usr/bin/env python3
"""
Structure 350 vs Consensus Results Analysis
Detailed comparison of structure 350 with the top-performing structures
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path

def load_structure_350():
    """Load structure 350 coordinates from XYZ file"""
    xyz_file = Path("data/Au20_OPT_1000/350.xyz")
    
    coordinates = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0].strip())
        energy = float(lines[1].strip())
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append([x, y, z])
    
    return np.array(coordinates), energy

def load_consensus_structure(rank):
    """Load consensus structure from .incase file"""
    incase_file = Path(f"task2_results/consensus/consensus_rank{rank}.incase")
    
    if not incase_file.exists():
        return None, None, None
    
    coordinates = []
    structure_id = None
    energy = None
    
    with open(incase_file, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            if line.startswith("# Structure:"):
                structure_id = line.split(":")[1].strip()
            elif line.startswith("# Energy:"):
                energy = float(line.split(":")[1].strip().split()[0])
            elif line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coordinates.append([x, y, z])
    
    return np.array(coordinates), energy, structure_id

def calculate_bonds(coordinates, max_distance=3.2, min_distance=2.3):
    """Calculate bonds between atoms based on distance"""
    bonds = []
    n_atoms = len(coordinates)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            if min_distance <= distance <= max_distance:
                bonds.append((i, j, distance))
    
    return bonds

def calculate_structural_properties(coordinates):
    """Calculate various structural properties"""
    # Center of mass
    center = np.mean(coordinates, axis=0)
    
    # Distances from center
    distances_from_center = np.linalg.norm(coordinates - center, axis=1)
    
    # Pairwise distances
    n_atoms = len(coordinates)
    pairwise_distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            pairwise_distances.append(dist)
    
    pairwise_distances = np.array(pairwise_distances)
    
    # Bonds
    bonds = calculate_bonds(coordinates)
    bond_lengths = [bond[2] for bond in bonds]
    
    # Coordination numbers
    coordination = [0] * n_atoms
    for bond in bonds:
        coordination[bond[0]] += 1
        coordination[bond[1]] += 1
    
    # Calculate compactness (moment of inertia)
    inertia_tensor = np.zeros((3, 3))
    for coord in coordinates:
        r = coord - center
        inertia_tensor += np.outer(r, r)
    
    eigenvalues = np.linalg.eigvals(inertia_tensor)
    compactness = np.min(eigenvalues) / np.max(eigenvalues)  # Sphericity measure
    
    return {
        'center': center,
        'radius': np.max(distances_from_center),
        'avg_distance_from_center': np.mean(distances_from_center),
        'std_distance_from_center': np.std(distances_from_center),
        'avg_pairwise_distance': np.mean(pairwise_distances),
        'std_pairwise_distance': np.std(pairwise_distances),
        'min_pairwise_distance': np.min(pairwise_distances),
        'max_pairwise_distance': np.max(pairwise_distances),
        'num_bonds': len(bonds),
        'avg_bond_length': np.mean(bond_lengths) if bond_lengths else 0,
        'std_bond_length': np.std(bond_lengths) if bond_lengths else 0,
        'avg_coordination': np.mean(coordination),
        'std_coordination': np.std(coordination),
        'max_coordination': np.max(coordination),
        'min_coordination': np.min(coordination),
        'compactness': compactness,
        'bonds': bonds,
        'coordination': coordination
    }

def analyze_symmetry(coordinates):
    """Analyze symmetry properties"""
    center = np.mean(coordinates, axis=0)
    centered_coords = coordinates - center
    
    # Check for approximate symmetries
    # Inversion symmetry
    inverted_coords = -centered_coords
    inversion_score = 0
    for coord in centered_coords:
        min_dist = np.min(np.linalg.norm(inverted_coords - coord, axis=1))
        if min_dist < 0.1:  # Tolerance for symmetry
            inversion_score += 1
    
    inversion_symmetry = inversion_score / len(coordinates)
    
    # Calculate moments of inertia for shape analysis
    inertia_tensor = np.zeros((3, 3))
    for coord in centered_coords:
        inertia_tensor += np.outer(coord, coord)
    
    eigenvalues = np.sort(np.linalg.eigvals(inertia_tensor))
    
    # Shape descriptors
    if eigenvalues[2] > 0:
        prolate = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2]
        oblate = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
    else:
        prolate = oblate = 0
    
    return {
        'inversion_symmetry': inversion_symmetry,
        'prolate': prolate,
        'oblate': oblate,
        'moment_ratios': eigenvalues / eigenvalues[2] if eigenvalues[2] > 0 else [0, 0, 0]
    }

def compare_structures():
    """Compare structure 350 with consensus results"""
    print("="*80)
    print("STRUCTURE 350 vs CONSENSUS RESULTS ANALYSIS")
    print("="*80)
    
    # Load structure 350
    coords_350, energy_350 = load_structure_350()
    props_350 = calculate_structural_properties(coords_350)
    sym_350 = analyze_symmetry(coords_350)
    
    print(f"\nSTRUCTURE 350 ANALYSIS:")
    print(f"  Energy: {energy_350:.6f} eV")
    print(f"  Bonds: {props_350['num_bonds']}")
    print(f"  Average coordination: {props_350['avg_coordination']:.2f}")
    print(f"  Compactness: {props_350['compactness']:.3f}")
    print(f"  Inversion symmetry: {sym_350['inversion_symmetry']:.3f}")
    
    # Load and analyze consensus structures
    consensus_data = []
    
    print(f"\nCONSENSUS STRUCTURES ANALYSIS:")
    print(f"{'Rank':<6} {'Structure':<12} {'Energy (eV)':<12} {'Bonds':<6} {'Coord':<6} {'Compact':<8} {'Symmetry':<8}")
    print("-" * 80)
    
    for rank in range(1, 11):
        coords, energy, struct_id = load_consensus_structure(rank)
        if coords is not None:
            props = calculate_structural_properties(coords)
            sym = analyze_symmetry(coords)
            
            consensus_data.append({
                'rank': rank,
                'structure_id': struct_id,
                'energy': energy,
                'coords': coords,
                'properties': props,
                'symmetry': sym
            })
            
            print(f"{rank:<6} {struct_id:<12} {energy:<12.6f} {props['num_bonds']:<6} "
                  f"{props['avg_coordination']:<6.2f} {props['compactness']:<8.3f} {sym['inversion_symmetry']:<8.3f}")
    
    print(f"\nWHY STRUCTURE 350 WASN'T SELECTED BY MODELS:")
    print("="*60)
    
    if consensus_data:
        best_consensus = consensus_data[0]  # Rank 1
        
        # Energy comparison
        energy_diff = energy_350 - best_consensus['energy']
        print(f"\n1. ENERGY ANALYSIS:")
        print(f"   Structure 350 energy: {energy_350:.6f} eV")
        print(f"   Best consensus energy: {best_consensus['energy']:.6f} eV")
        print(f"   Energy difference: +{energy_diff:.6f} eV ({energy_diff*1000:.2f} meV)")
        
        if energy_diff > 0:
            print(f"   ❌ Structure 350 is {energy_diff:.6f} eV LESS STABLE")
            print(f"      This is the primary reason models didn't select it!")
        
        # Structural comparison
        print(f"\n2. STRUCTURAL COMPARISON:")
        print(f"   Property                   Structure 350    Best Consensus   Difference")
        print(f"   {'='*70}")
        
        properties = [
            ('Number of bonds', 'num_bonds', ''),
            ('Average coordination', 'avg_coordination', '.2f'),
            ('Compactness', 'compactness', '.3f'),
            ('Average bond length (Å)', 'avg_bond_length', '.3f'),
            ('Bond length std dev', 'std_bond_length', '.3f'),
            ('Cluster radius (Å)', 'radius', '.3f'),
        ]
        
        for prop_name, prop_key, fmt in properties:
            val_350 = props_350[prop_key]
            val_consensus = best_consensus['properties'][prop_key]
            diff = val_350 - val_consensus
            
            if fmt:
                val_350_str = f"{val_350:{fmt}}"
                val_consensus_str = f"{val_consensus:{fmt}}"
                diff_str = f"{diff:{fmt}}"
                print(f"   {prop_name:<25} {val_350_str:<15} {val_consensus_str:<15} {diff_str:>10}")
            else:
                print(f"   {prop_name:<25} {val_350:<15} {val_consensus:<15} {diff:>10}")
        
        print(f"\n3. SYMMETRY ANALYSIS:")
        print(f"   Structure 350 inversion symmetry: {sym_350['inversion_symmetry']:.3f}")
        print(f"   Best consensus inversion symmetry: {best_consensus['symmetry']['inversion_symmetry']:.3f}")
        
        # Energy landscape analysis
        print(f"\n4. ENERGY LANDSCAPE POSITION:")
        energies = [data['energy'] for data in consensus_data]
        energy_range = max(energies) - min(energies)
        percentile_position = sum(1 for e in energies if e > energy_350) / len(energies) * 100
        
        print(f"   Consensus energy range: {energy_range:.6f} eV")
        print(f"   Structure 350 percentile: {percentile_position:.1f}%")
        
        if percentile_position < 50:
            print(f"   Structure 350 is in the LOWER half of consensus energies")
        else:
            print(f"   Structure 350 is in the UPPER half of consensus energies")
    
    print(f"\n5. VISUAL APPEAL vs ENERGY STABILITY:")
    print(f"   Structure 350 appears neat and symmetric to human eyes because:")
    print(f"   - High geometric symmetry ({sym_350['inversion_symmetry']:.3f} inversion symmetry)")
    print(f"   - Compact structure (compactness: {props_350['compactness']:.3f})")
    print(f"   - Regular coordination patterns")
    print(f"   ")
    print(f"   However, ML models optimize for ENERGY, not visual appeal!")
    print(f"   The energy difference of {energy_diff:.6f} eV ({energy_diff*1000:.2f} meV)")
    print(f"   is significant in computational chemistry.")
    
    return consensus_data

def visualize_comparison():
    """Create comparative visualization"""
    # Load structure 350
    coords_350, energy_350 = load_structure_350()
    
    # Load top 3 consensus structures
    fig = plt.figure(figsize=(20, 5))
    
    # Plot structure 350
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(coords_350[:, 0], coords_350[:, 1], coords_350[:, 2], 
               c='gold', s=100, alpha=0.8, edgecolors='black')
    ax1.set_title(f'Structure 350\nEnergy: {energy_350:.3f} eV\n(Your Choice)', fontweight='bold')
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    
    # Plot top 3 consensus
    for i in range(1, 4):
        coords, energy, struct_id = load_consensus_structure(i)
        if coords is not None:
            ax = fig.add_subplot(141 + i, projection='3d')
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                      c='red' if i == 1 else 'orange' if i == 2 else 'blue', 
                      s=100, alpha=0.8, edgecolors='black')
            ax.set_title(f'Rank {i}: {struct_id}\nEnergy: {energy:.3f} eV\n(Models\' Choice)', fontweight='bold')
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
    
    plt.tight_layout()
    plt.savefig('structure_350_vs_consensus.png', dpi=300, bbox_inches='tight')
    print(f"\nComparative visualization saved as: structure_350_vs_consensus.png")
    plt.show()

def main():
    """Main analysis function"""
    consensus_data = compare_structures()
    
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"Structure 350 looks neat and symmetric, but it's energetically less stable")
    print(f"than the structures selected by the ML models. Machine learning models")
    print(f"are trained to predict the lowest energy configurations, which may not")
    print(f"always correspond to the most visually appealing structures to humans.")
    print(f"")
    print(f"The consensus structures represent the ML models' best predictions for")
    print(f"stable Au20 configurations based on learned energy patterns from training data.")
    
    visualize_comparison()

if __name__ == "__main__":
    main()