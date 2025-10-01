#!/usr/bin/env python3
"""
Structure 350 Visualization and Stability Analysis
Analyzes Au20 structure 350 and compares it with consensus results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path

def load_structure_350():
    """Load structure 350 coordinates from XYZ file"""
    xyz_file = Path("data/Au20_OPT_1000/960.xyz")
    
    if not xyz_file.exists():
        print(f"Error: {xyz_file} not found")
        return None, None
    
    coordinates = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0].strip())
        energy = float(lines[1].strip())
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append([element, x, y, z])
    
    return coordinates, energy

def calculate_bonds(coordinates, max_distance=3.2, min_distance=2.3):
    """Calculate bonds between Au atoms based on distance"""
    bonds = []
    n_atoms = len(coordinates)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            x1, y1, z1 = coordinates[i][1:4]
            x2, y2, z2 = coordinates[j][1:4]
            
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            if min_distance <= distance <= max_distance:
                bonds.append((i, j, distance))
    
    return bonds

def analyze_structure_stability():
    """Analyze how structure 350 compares to consensus results"""
    print("="*60)
    print("STRUCTURE 350 STABILITY ANALYSIS")
    print("="*60)
    
    # Load structure 350
    coords, energy = load_structure_350()
    if coords is None:
        return
    
    print(f"\nStructure 350 Details:")
    print(f"  Energy: {energy:.6f} eV")
    print(f"  Atoms: {len(coords)} Au atoms")
    
    # Calculate bonds
    bonds = calculate_bonds(coords)
    print(f"  Bonds: {len(bonds)} Au-Au bonds")
    
    # Try to load consensus results for comparison
    consensus_files = [
        "task2_results/summary_report.md",
        "linear_models_results/top_20_stable_structures.csv",
        "tree_models_results/top_20_stable_structures.csv",
        "kernel_models_analysis/top_20_stable_structures_summary.csv"
    ]
    
    print(f"\nStability Comparison:")
    
    # Load available CSV results
    comparison_energies = []
    found_results = False
    
    for csv_file in consensus_files:
        if Path(csv_file).exists() and csv_file.endswith('.csv'):
            try:
                df = pd.read_csv(csv_file)
                if 'actual_energy' in df.columns:
                    energies = df['actual_energy'].values
                    comparison_energies.extend(energies)
                    found_results = True
                    print(f"  Loaded {len(energies)} structures from {Path(csv_file).name}")
            except Exception as e:
                print(f"  Could not load {csv_file}: {e}")
    
    if found_results:
        comparison_energies = np.array(comparison_energies)
        min_energy = np.min(comparison_energies)
        max_energy = np.max(comparison_energies)
        mean_energy = np.mean(comparison_energies)
        
        print(f"\nComparison with Model Results:")
        print(f"  Structure 350 energy: {energy:.6f} eV")
        print(f"  Best model energy:    {min_energy:.6f} eV")
        print(f"  Worst model energy:   {max_energy:.6f} eV")
        print(f"  Average model energy: {mean_energy:.6f} eV")
        
        energy_diff = energy - min_energy
        print(f"\nStability Assessment:")
        if energy_diff < 0:
            print(f"  ✅ Structure 350 is MORE STABLE than all model predictions!")
            print(f"     Energy difference: {abs(energy_diff):.6f} eV lower")
        elif energy_diff < 0.1:
            print(f"  ✅ Structure 350 is VERY STABLE (within 0.1 eV of best)")
            print(f"     Energy difference: +{energy_diff:.6f} eV")
        elif energy_diff < 0.5:
            print(f"  ⚠️  Structure 350 is MODERATELY STABLE")
            print(f"     Energy difference: +{energy_diff:.6f} eV above best")
        else:
            print(f"  ❌ Structure 350 is LESS STABLE")
            print(f"     Energy difference: +{energy_diff:.6f} eV above best")
        
        # Percentile ranking
        better_count = np.sum(comparison_energies > energy)
        percentile = (better_count / len(comparison_energies)) * 100
        print(f"  Percentile ranking: {percentile:.1f}% (better than {percentile:.1f}% of predicted structures)")
    
    else:
        print("  No comparison data available. Run model analysis first.")
    
    return coords, energy, bonds

def visualize_structure_350():
    """Create 3D visualization of structure 350"""
    coords, energy, bonds = analyze_structure_stability()
    
    if coords is None:
        return
    
    # Extract coordinates
    x_coords = [coord[1] for coord in coords]
    y_coords = [coord[2] for coord in coords]
    z_coords = [coord[3] for coord in coords]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c='gold', s=200, alpha=0.8, edgecolors='darkgoldenrod', linewidth=2)
    
    # Plot bonds
    for bond in bonds:
        i, j, distance = bond
        x_bond = [x_coords[i], x_coords[j]]
        y_bond = [y_coords[i], y_coords[j]]
        z_bond = [z_coords[i], z_coords[j]]
        
        ax.plot(x_bond, y_bond, z_bond, 'gray', alpha=0.6, linewidth=1.5)
    
    # Customize the plot
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(f'Au20 Structure 350\nEnergy: {energy:.6f} eV | Bonds: {len(bonds)}', 
                fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    max_range = max(max(x_coords) - min(x_coords),
                   max(y_coords) - min(y_coords),
                   max(z_coords) - min(z_coords)) / 2.0
    
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add some styling
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add info text
    info_text = f"""Structure Details:
• 20 Au atoms
• {len(bonds)} Au-Au bonds
• Energy: {energy:.6f} eV
• Bond range: 2.3-3.2 Å"""
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "structure_350_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n3D visualization saved as: {output_file}")
    
    # Show the plot
    plt.show()

def create_detailed_analysis():
    """Create detailed structural analysis"""
    coords, energy, bonds = analyze_structure_stability()
    
    if coords is None:
        return
    
    print(f"\n" + "="*60)
    print("DETAILED STRUCTURAL ANALYSIS")
    print("="*60)
    
    # Bond length distribution
    bond_lengths = [bond[2] for bond in bonds]
    
    print(f"\nBond Statistics:")
    print(f"  Total bonds: {len(bonds)}")
    print(f"  Average bond length: {np.mean(bond_lengths):.3f} Å")
    print(f"  Shortest bond: {np.min(bond_lengths):.3f} Å")
    print(f"  Longest bond: {np.max(bond_lengths):.3f} Å")
    print(f"  Bond length std dev: {np.std(bond_lengths):.3f} Å")
    
    # Coordination numbers
    coordination = [0] * len(coords)
    for bond in bonds:
        coordination[bond[0]] += 1
        coordination[bond[1]] += 1
    
    print(f"\nCoordination Analysis:")
    print(f"  Average coordination: {np.mean(coordination):.2f}")
    print(f"  Coordination range: {np.min(coordination)} - {np.max(coordination)}")
    
    coord_distribution = {}
    for coord in coordination:
        coord_distribution[coord] = coord_distribution.get(coord, 0) + 1
    
    print(f"  Coordination distribution:")
    for coord, count in sorted(coord_distribution.items()):
        print(f"    {coord}-coordinated atoms: {count}")
    
    # Geometric properties
    x_coords = [coord[1] for coord in coords]
    y_coords = [coord[2] for coord in coords]
    z_coords = [coord[3] for coord in coords]
    
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    center_z = np.mean(z_coords)
    
    # Calculate distances from center
    distances_from_center = []
    for coord in coords:
        dist = np.sqrt((coord[1] - center_x)**2 + (coord[2] - center_y)**2 + (coord[3] - center_z)**2)
        distances_from_center.append(dist)
    
    print(f"\nGeometric Properties:")
    print(f"  Cluster center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    print(f"  Average distance from center: {np.mean(distances_from_center):.3f} Å")
    print(f"  Cluster radius (max distance): {np.max(distances_from_center):.3f} Å")
    print(f"  Cluster compactness: {np.std(distances_from_center):.3f} Å")

def main():
    """Main function"""
    print("Structure 350 Analysis and Visualization")
    print("="*60)
    
    # Run complete analysis
    create_detailed_analysis()
    
    # Create visualization
    print(f"\nGenerating 3D visualization...")
    visualize_structure_350()
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()