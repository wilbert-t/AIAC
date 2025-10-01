#!/usr/bin/env python3
"""
Au20 Cluster Structure Analysis - Multi-Category Comparison with Consensus Visualization
Combines linear, kernel, and tree models to find the most stable Au20 structures

Expected Directory Structure:
- linear_models_results/top_20_stable_structures.csv
- kernel_models_analysis/top_20_stable_structures.csv  
- tree_models_results/top_20_stable_structures.csv

Run the individual model scripts first to generate these CSV files:
1. python 1.linear_models.py
2. python 2.kernel_models.py
3. python 3.tree_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Model category definitions
MODEL_CATEGORIES = {
    'linear': ['svr_linear', 'elastic_net', 'ridge', 'lasso'],
    'kernel': ['svr_rbf_conservative'],  
    'tree': ['knn_stable', 'random_forest', 'xgboost', 'lightgbm', 
             'catboost', 'extra_trees', 'gradient_boosting']
}

class Au20StructureAnalyzer:
    """Analyzer for finding the most stable Au20 structures across model categories"""
    
    def __init__(self, csv_paths: Dict[str, str], output_dir: str = './outputs'):
        """
        Args:
            csv_paths: Dict with keys 'linear', 'kernel', 'tree' pointing to CSV files
            output_dir: Output directory for results
        """
        self.csv_paths = {k: Path(v) for k, v in csv_paths.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.category_dfs = {}
        self.combined_df = None
        self.category_results = {}
        self.consensus_structures = []
        
    def load_data(self):
        """Load and validate CSV data from all categories"""
        print("Loading data from all categories...")
        
        all_dfs = []
        for category, csv_path in self.csv_paths.items():
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found, skipping {category} category")
                continue
            
            df = pd.read_csv(csv_path)
            df = df[df['n_atoms'] == 20].copy()  # Filter for Au20 only
            df['category'] = category
            
            self.category_dfs[category] = df
            all_dfs.append(df)
            
            print(f"  {category}: {len(df)} Au20 structures from {df['model_name'].nunique()} models")
        
        if not all_dfs:
            raise ValueError("No valid CSV files found!")
        
        # Combine all data
        self.combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal: {len(self.combined_df)} Au20 structures across all categories")
        
    def parse_coordinates(self, row) -> List[Tuple[str, float, float, float]]:
        """Extract atomic coordinates from flattened CSV row"""
        coords = []
        atom_idx = 1
        
        while f'atom_{atom_idx}_element' in row.index:
            if pd.notna(row[f'atom_{atom_idx}_element']):
                element = row[f'atom_{atom_idx}_element']
                x = row[f'atom_{atom_idx}_x']
                y = row[f'atom_{atom_idx}_y']
                z = row[f'atom_{atom_idx}_z']
                coords.append((element, x, y, z))
            atom_idx += 1
            
        return coords
    
    def analyze_category(self, category_name: str) -> Dict:
        """Analyze all models within a category and find the best structure"""
        print(f"\n{'='*60}")
        print(f"Analyzing {category_name.upper()} category")
        print(f"{'='*60}")
        
        if category_name not in self.category_dfs:
            print(f"  No data available for {category_name}")
            return None
        
        df = self.category_dfs[category_name]
        category_dir = self.output_dir / category_name
        category_dir.mkdir(exist_ok=True)
        
        # Find best structure for each model
        model_results = {}
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            
            # Sort by actual_energy (lowest = most stable)
            best_row = model_data.sort_values('actual_energy').iloc[0]
            
            model_results[model_name] = {
                'structure_id': best_row['structure_id'],
                'actual_energy': best_row['actual_energy'],
                'predicted_energy': best_row['predicted_energy'],
                'n_atoms': best_row['n_atoms'],
                'row': best_row
            }
            
            print(f"  {model_name}: {best_row['structure_id']} (Energy: {best_row['actual_energy']:.6f} eV)")
        
        # Find overall best in category
        best_model = min(model_results.items(), key=lambda x: x[1]['actual_energy'])
        best_model_name, best_data = best_model
        
        print(f"\nCategory winner: {best_model_name}")
        print(f"   Structure: {best_data['structure_id']}")
        print(f"   Energy: {best_data['actual_energy']:.6f} eV")
        
        # Generate output files
        self.create_incase_file(category_dir, category_name, best_data)
        self.create_html_visualization(category_dir, category_name, best_model_name, best_data)
        
        return {
            'category': category_name,
            'best_model': best_model_name,
            'structure_id': best_data['structure_id'],
            'actual_energy': best_data['actual_energy'],
            'predicted_energy': best_data['predicted_energy'],
            'all_models': model_results,
            'best_row': best_data['row']
        }
    
    def find_consensus_structures(self, top_n: int = 10):
        """Find the most commonly predicted structures across all models"""
        print(f"\n{'='*60}")
        print(f"Finding Top {top_n} Consensus Structures")
        print(f"{'='*60}")
        
        # Count structure occurrences across all models
        structure_votes = Counter()
        structure_energies = {}
        structure_rows = {}
        
        for _, row in self.combined_df.iterrows():
            struct_id = row['structure_id']
            structure_votes[struct_id] += 1
            
            # Keep track of best energy for each structure
            if struct_id not in structure_energies or row['actual_energy'] < structure_energies[struct_id]:
                structure_energies[struct_id] = row['actual_energy']
                structure_rows[struct_id] = row
        
        # Get top N most common structures
        most_common = structure_votes.most_common(top_n)
        
        print(f"\nTop {top_n} most frequently predicted structures:\n")
        print(f"{'Rank':<6} {'Structure ID':<15} {'Votes':<8} {'Energy (eV)':<15} {'Models'}")
        print("-" * 70)
        
        consensus_data = []
        for rank, (struct_id, count) in enumerate(most_common, 1):
            energy = structure_energies[struct_id]
            
            # Find which models predicted this structure
            models = self.combined_df[self.combined_df['structure_id'] == struct_id]['model_name'].unique()
            models_str = ', '.join(models[:3]) + ('...' if len(models) > 3 else '')
            
            print(f"{rank:<6} {struct_id:<15} {count:<8} {energy:<15.6f} {models_str}")
            
            consensus_data.append({
                'rank': rank,
                'structure_id': struct_id,
                'votes': count,
                'energy': energy,
                'models': list(models),
                'row': structure_rows[struct_id]
            })
        
        self.consensus_structures = consensus_data
        return consensus_data
    
    def create_consensus_visualization(self):
        """Create combined HTML visualization of top 10 consensus structures"""
        print(f"\nCreating consensus visualization...")
        
        consensus_dir = self.output_dir / 'consensus'
        consensus_dir.mkdir(exist_ok=True)
        
        if not self.consensus_structures:
            print("  No consensus structures to visualize")
            return
        
        # Create individual files for each structure
        for data in self.consensus_structures:
            self.create_incase_file(
                consensus_dir, 
                f"consensus_rank{data['rank']}", 
                {'row': data['row'], 'structure_id': data['structure_id'], 
                 'actual_energy': data['energy'], 'n_atoms': 20}
            )
        
        # Create combined multi-structure HTML
        filename = consensus_dir / 'consensus_top10.html'
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top 10 Consensus Au20 Structures</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            font-size: 36px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 18px;
            margin-top: 10px;
            opacity: 0.9;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .structure-card {
            background: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        .structure-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
        }
        .structure-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .rank {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .rank.top3 {
            color: #ffd700;
        }
        .structure-info {
            flex-grow: 1;
            margin-left: 15px;
        }
        .structure-id {
            font-weight: bold;
            font-size: 16px;
            color: #333;
        }
        .votes {
            color: #667eea;
            font-weight: bold;
        }
        .energy {
            color: #d32f2f;
            font-size: 14px;
        }
        .canvas-container {
            width: 100%;
            height: 300px;
            position: relative;
        }
        canvas {
            display: block;
            cursor: grab;
        }
        canvas:active {
            cursor: grabbing;
        }
        .models {
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Top 10 Consensus Au20 Structures</h1>
        <div class="subtitle">Most frequently predicted stable structures across all models</div>
    </div>
    
    <div class="grid" id="structures-grid"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
"""
        
        # Add structure data
        structures_js = []
        for data in self.consensus_structures[:10]:
            coords = self.parse_coordinates(data['row'])
            coord_list = [[c[1], c[2], c[3]] for c in coords]
            
            # Calculate bonds
            bonds = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = np.sqrt(
                        (coords[i][1] - coords[j][1])**2 +
                        (coords[i][2] - coords[j][2])**2 +
                        (coords[i][3] - coords[j][3])**2
                    )
                    if 2.3 <= dist <= 3.2:
                        bonds.append([i, j])
            
            structures_js.append({
                'rank': data['rank'],
                'structure_id': data['structure_id'],
                'votes': data['votes'],
                'energy': f"{data['energy']:.6f}",
                'models': data['models'][:3],
                'atoms': coord_list,
                'bonds': bonds
            })
        
        html_content += f"        const structures = {str(structures_js).replace('True', 'true').replace('False', 'false')};\n"
        
        html_content += """
        // Create structure cards
        const grid = document.getElementById('structures-grid');
        
        structures.forEach((struct, idx) => {
            const card = document.createElement('div');
            card.className = 'structure-card';
            
            const rankClass = struct.rank <= 3 ? 'top3' : '';
            const medal = struct.rank === 1 ? '🥇' : struct.rank === 2 ? '🥈' : struct.rank === 3 ? '🥉' : '';
            
            card.innerHTML = `
                <div class="structure-header">
                    <div class="rank ${rankClass}">${medal} #${struct.rank}</div>
                    <div class="structure-info">
                        <div class="structure-id">${struct.structure_id}</div>
                        <div class="votes">${struct.votes} votes</div>
                        <div class="energy">Energy: ${struct.energy} eV</div>
                    </div>
                </div>
                <div class="canvas-container" id="canvas-${idx}"></div>
                <div class="models">Models: ${struct.models.join(', ')}${struct.models.length > 3 ? '...' : ''}</div>
            `;
            
            grid.appendChild(card);
            
            // Create 3D visualization for this structure
            createVisualization(struct, idx);
        });
        
        function createVisualization(struct, idx) {
            const container = document.getElementById(`canvas-${idx}`);
            const width = container.clientWidth;
            const height = 300;
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf5f5f5);
            
            const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
            camera.position.set(12, 12, 12);
            
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);
            
            // Materials
            const atomMaterial = new THREE.MeshPhongMaterial({
                color: 0xffd700,
                shininess: 100,
                specular: 0xffffff
            });
            
            const bondMaterial = new THREE.MeshBasicMaterial({
                color: 0x808080,
                transparent: true,
                opacity: 0.5
            });
            
            // Create atoms
            const atomGeometry = new THREE.SphereGeometry(0.35, 24, 24);
            struct.atoms.forEach(pos => {
                const atom = new THREE.Mesh(atomGeometry, atomMaterial);
                atom.position.set(pos[0], pos[1], pos[2]);
                scene.add(atom);
            });
            
            // Create bonds
            struct.bonds.forEach(bond => {
                const start = new THREE.Vector3(...struct.atoms[bond[0]]);
                const end = new THREE.Vector3(...struct.atoms[bond[1]]);
                const direction = new THREE.Vector3().subVectors(end, start);
                const length = direction.length();
                
                const bondGeometry = new THREE.CylinderGeometry(0.06, 0.06, length, 6);
                const bondMesh = new THREE.Mesh(bondGeometry, bondMaterial);
                
                bondMesh.position.copy(start.clone().add(direction.multiplyScalar(0.5)));
                bondMesh.lookAt(end);
                bondMesh.rotateX(Math.PI / 2);
                
                scene.add(bondMesh);
            });
            
            // Mouse controls
            let isDragging = false;
            let previousMouse = { x: 0, y: 0 };
            
            renderer.domElement.addEventListener('mousedown', (e) => {
                isDragging = true;
                previousMouse = { x: e.clientX, y: e.clientY };
            });
            
            renderer.domElement.addEventListener('mouseup', () => {
                isDragging = false;
            });
            
            renderer.domElement.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const deltaX = e.clientX - previousMouse.x;
                    const deltaY = e.clientY - previousMouse.y;
                    
                    camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), deltaX * 0.01);
                    camera.position.applyAxisAngle(
                        new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion),
                        deltaY * 0.01
                    );
                    camera.lookAt(scene.position);
                }
                previousMouse = { x: e.clientX, y: e.clientY };
            });
            
            renderer.domElement.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomSpeed = 0.1;
                const direction = new THREE.Vector3();
                camera.getWorldDirection(direction);
                camera.position.addScaledVector(direction, e.deltaY * zoomSpeed);
            });
            
            // Animation
            function animate() {
                requestAnimationFrame(animate);
                camera.lookAt(scene.position);
                renderer.render(scene, camera);
            }
            animate();
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            location.reload(); // Simple solution for responsive grid
        });
    </script>
</body>
</html>"""
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"  Created {filename.name}")
    
    def create_incase_file(self, category_dir: Path, prefix: str, data: Dict):
        """Create .incase file with structure coordinates"""
        coords = self.parse_coordinates(data['row'])
        
        filename = category_dir / f'{prefix}.incase'
        
        with open(filename, 'w') as f:
            f.write(f"# Structure: {data['structure_id']}\n")
            f.write(f"# Energy: {data.get('actual_energy', 'N/A')} eV\n")
            f.write(f"# Number of atoms: {data.get('n_atoms', len(coords))}\n")
            f.write("\n")
            
            for element, x, y, z in coords:
                f.write(f"{element:3s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
    
    def create_html_visualization(self, category_dir: Path, category_name: str, 
                                  model_name: str, data: Dict):
        """Create 3D HTML visualization using Three.js"""
        coords = self.parse_coordinates(data['row'])
        
        # Calculate bonds
        bonds = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.sqrt(
                    (coords[i][1] - coords[j][1])**2 +
                    (coords[i][2] - coords[j][2])**2 +
                    (coords[i][3] - coords[j][3])**2
                )
                if 2.3 <= dist <= 3.2:
                    bonds.append((i, j))
        
        filename = category_dir / f'{category_name}_best.html'
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{category_name.upper()} - Au20 Structure</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            overflow: hidden;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            z-index: 100;
            max-width: 350px;
        }}
        h1 {{
            margin: 0 0 15px 0;
            color: #2a5298;
            font-size: 24px;
        }}
        .info-row {{
            margin: 8px 0;
            font-size: 14px;
            color: #333;
        }}
        .label {{
            font-weight: bold;
            color: #1e3c72;
        }}
        .energy {{
            color: #d32f2f;
            font-weight: bold;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
        }}
        .controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            font-size: 13px;
            color: #555;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h1>{category_name.upper()} Category Winner</h1>
        <div class="info-row"><span class="label">Model:</span> {model_name}</div>
        <div class="info-row"><span class="label">Structure:</span> {data['structure_id']}</div>
        <div class="info-row"><span class="label">Energy:</span> <span class="energy">{data['actual_energy']:.6f} eV</span></div>
        <div class="info-row"><span class="label">Atoms:</span> {data['n_atoms']} Au atoms</div>
        <div class="info-row"><span class="label">Bonds:</span> {len(bonds)} Au-Au bonds</div>
    </div>
    
    <div class="controls">
        Drag to rotate | Scroll to zoom | Right-drag to pan
    </div>
    
    <div id="container"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(15, 15, 15);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('container').appendChild(renderer.domElement);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(10, 10, 10);
        scene.add(directionalLight1);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-10, -10, -10);
        scene.add(directionalLight2);
        
        const atomMaterial = new THREE.MeshPhongMaterial({{
            color: 0xffd700,
            shininess: 100,
            specular: 0xffffff
        }});
        
        const bondMaterial = new THREE.MeshBasicMaterial({{
            color: 0x808080,
            transparent: true,
            opacity: 0.6
        }});
        
        const atoms = {str([[c[1], c[2], c[3]] for c in coords])};
        const bonds = {str(bonds)};
        
        const atomGeometry = new THREE.SphereGeometry(0.4, 32, 32);
        atoms.forEach(pos => {{
            const atom = new THREE.Mesh(atomGeometry, atomMaterial);
            atom.position.set(pos[0], pos[1], pos[2]);
            scene.add(atom);
        }});
        
        bonds.forEach(bond => {{
            const start = new THREE.Vector3(...atoms[bond[0]]);
            const end = new THREE.Vector3(...atoms[bond[1]]);
            const direction = new THREE.Vector3().subVectors(end, start);
            const length = direction.length();
            
            const bondGeometry = new THREE.CylinderGeometry(0.08, 0.08, length, 8);
            const bondMesh = new THREE.Mesh(bondGeometry, bondMaterial);
            
            bondMesh.position.copy(start.clone().add(direction.multiplyScalar(0.5)));
            bondMesh.lookAt(end);
            bondMesh.rotateX(Math.PI / 2);
            
            scene.add(bondMesh);
        }});
        
        let isDragging = false;
        let isPanning = false;
        let previousMousePosition = {{ x: 0, y: 0 }};
        
        renderer.domElement.addEventListener('mousedown', (e) => {{
            if (e.button === 0) isDragging = true;
            if (e.button === 2) isPanning = true;
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
        }});
        
        renderer.domElement.addEventListener('mouseup', () => {{
            isDragging = false;
            isPanning = false;
        }});
        
        renderer.domElement.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), deltaX * 0.005);
                camera.position.applyAxisAngle(
                    new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion),
                    deltaY * 0.005
                );
                camera.lookAt(scene.position);
            }}
            
            if (isPanning) {{
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                camera.position.x -= deltaX * 0.02;
                camera.position.y += deltaY * 0.02;
            }}
            
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
        }});
        
        renderer.domElement.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const direction = new THREE.Vector3();
            camera.getWorldDirection(direction);
            camera.position.addScaledVector(direction, e.deltaY * 0.1);
        }});
        
        renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
        
        function animate() {{
            requestAnimationFrame(animate);
            camera.lookAt(scene.position);
            renderer.render(scene, camera);
        }}
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        animate();
    </script>
</body>
</html>"""
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"  Created {filename.name}")
    
    def generate_summary_report(self):
        """Generate comprehensive markdown summary"""
        print(f"\n{'='*60}")
        print("Generating Summary Report")
        print(f"{'='*60}")
        
        valid_results = {k: v for k, v in self.category_results.items() if v is not None}
        
        if not valid_results:
            print("  No valid results to summarize")
            return
        
        overall_best = min(valid_results.items(), key=lambda x: x[1]['actual_energy'])
        best_category, best_data = overall_best
        
        report = f"""# Au20 Gold Cluster Stability Analysis
## Comprehensive Multi-Category Comparison

**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overall Winner

**Most Stable Au20 Structure:** `{best_data['structure_id']}`

- **Category:** {best_category.upper()}
- **Model:** {best_data['best_model']}
- **Actual Energy:** {best_data['actual_energy']:.6f} eV
- **Predicted Energy:** {best_data['predicted_energy']:.6f} eV

**Files:**
- `.incase` file: `{best_category}/{best_category}_best.incase`
- 3D visualization: `{best_category}/{best_category}_best.html`

---

## Category Results

"""
        
        for category_name in ['linear', 'kernel', 'tree']:
            result = self.category_results.get(category_name)
            
            if result is None:
                report += f"### {category_name.upper()} Category\n\n"
                report += "No structures found for this category\n\n"
                continue
            
            is_winner = category_name == best_category
            winner_badge = "🥇 " if is_winner else ""
            
            report += f"### {winner_badge}{category_name.upper()} Category\n\n"
            report += f"**Best Model:** {result['best_model']}\n\n"
            report += f"**Structure:** {result['structure_id']}\n\n"
            report += f"**Energy:** {result['actual_energy']:.6f} eV\n\n"
            
            if len(result['all_models']) > 1:
                report += "**Models in Category:**\n\n"
                report += "| Model | Structure | Energy (eV) |\n"
                report += "|-------|-----------|-------------|\n"
                
                sorted_models = sorted(result['all_models'].items(), 
                                     key=lambda x: x[1]['actual_energy'])
                
                for model_name, model_data in sorted_models:
                    energy = model_data['actual_energy']
                    struct_id = model_data['structure_id']
                    report += f"| {model_name} | {struct_id} | {energy:.6f} |\n"
                
                report += "\n"
            
            report += f"**Files:**\n"
            report += f"- `{category_name}/{category_name}_best.incase`\n"
            report += f"- `{category_name}/{category_name}_best.html`\n\n"
        
        report += "---\n\n## Cross-Category Energy Comparison\n\n"
        report += "| Rank | Category | Model | Structure | Energy (eV) |\n"
        report += "|------|----------|-------|-----------|-------------|\n"
        
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]['actual_energy'])
        
        for rank, (cat, data) in enumerate(sorted_results, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            report += f"| {medal} | {cat.upper()} | {data['best_model']} | "
            report += f"{data['structure_id']} | {data['actual_energy']:.6f} |\n"
        
        if len(sorted_results) > 1:
            energy_range = sorted_results[-1][1]['actual_energy'] - sorted_results[0][1]['actual_energy']
            report += f"\n**Energy Range:** {energy_range:.6f} eV\n\n"
        
        report += "---\n\n## Top 10 Consensus Structures\n\n"
        report += "Structures most frequently predicted as stable across all models:\n\n"
        report += "| Rank | Structure | Votes | Energy (eV) | Primary Models |\n"
        report += "|------|-----------|-------|-------------|----------------|\n"
        
        for cons in self.consensus_structures[:10]:
            models_str = ', '.join(cons['models'][:2])
            if len(cons['models']) > 2:
                models_str += f" (+{len(cons['models'])-2} more)"
            report += f"| {cons['rank']} | {cons['structure_id']} | {cons['votes']} | "
            report += f"{cons['energy']:.6f} | {models_str} |\n"
        
        report += f"\n**Consensus Visualization:** `consensus/consensus_top10.html`\n\n"
        
        report += "---\n\n## Key Findings\n\n"
        
        if self.consensus_structures:
            top_consensus = self.consensus_structures[0]
            report += f"1. **Most Agreed Upon Structure:** `{top_consensus['structure_id']}` "
            report += f"({top_consensus['votes']} models agree)\n\n"
        
        report += f"2. **Lowest Energy Structure:** `{best_data['structure_id']}` "
        report += f"({best_data['actual_energy']:.6f} eV)\n\n"
        
        if self.consensus_structures[0]['structure_id'] == best_data['structure_id']:
            report += "   **Note:** The most stable structure is also the most frequently predicted!\n\n"
        else:
            report += f"   **Note:** Different from most frequent prediction. Energy difference: "
            energy_diff = abs(self.consensus_structures[0]['energy'] - best_data['actual_energy'])
            report += f"{energy_diff:.6f} eV\n\n"
        
        report += f"3. **Total Structures Analyzed:** {len(self.combined_df)} Au20 clusters\n\n"
        report += f"4. **Total Models:** {self.combined_df['model_name'].nunique()} across 3 categories\n\n"
        
        report += "---\n\n## Recommendations\n\n"
        
        report += f"1. **For highest accuracy:** Use `{best_data['structure_id']}` from "
        report += f"{best_category.upper()} ({best_data['best_model']})\n\n"
        
        report += f"2. **For consensus approach:** Consider the top 3 consensus structures "
        report += "which show agreement across multiple models\n\n"
        
        report += "3. **Visualization:** Open HTML files in a web browser:\n"
        report += "   - Individual category winners: `{category}/{category}_best.html`\n"
        report += "   - Top 10 consensus: `consensus/consensus_top10.html`\n\n"
        
        report += "4. **Further analysis:** Use `.incase` files for computational chemistry software\n\n"
        
        report_path = self.output_dir / 'summary_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Created {report_path.name}")
        print(f"\n{report}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("AU20 CLUSTER MULTI-CATEGORY ANALYSIS")
        print("="*60)
        
        self.load_data()
        
        for category_name in ['linear', 'kernel', 'tree']:
            result = self.analyze_category(category_name)
            self.category_results[category_name] = result
        
        self.find_consensus_structures(top_n=10)
        self.create_consensus_visualization()
        self.generate_summary_report()
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  📄 summary_report.md")
        print("  📁 linear/")
        print("  📁 kernel/")
        print("  📁 tree/")
        print("  📁 consensus/ (top 10 structures)")
        print("\n💡 Open .html files in browser for 3D visualization")


def main():
    """Main entry point"""
    print("Au20 Structure Analysis - Multi-Category")
    print("="*60)
    
    # Hardcoded CSV paths - automatically find the most stable structures CSV files
    csv_paths = {}
    
    # Expected directory structure and file patterns
    category_dirs = {
        'linear': 'linear_models_results',
        'kernel': 'kernel_models_analysis', 
        'tree': 'tree_models_results'
    }
    
    print("\nLooking for CSV files in category directories...")
    
    for category, dir_name in category_dirs.items():
        dir_path = Path(dir_name)
        
        if not dir_path.exists():
            print(f"  ❌ {category}: Directory {dir_name} not found")
            continue
            
        # Look for the top structures CSV file
        possible_files = [
            dir_path / 'top_20_stable_structures.csv',
            dir_path / 'top_20_stable_structures_summary.csv',
            dir_path / f'{category}_top_20_stable_structures.csv'
        ]
        
        found_file = None
        for file_path in possible_files:
            if file_path.exists():
                found_file = file_path
                break
        
        if found_file:
            csv_paths[category] = str(found_file)
            print(f"  ✅ {category}: Found {found_file.name}")
        else:
            print(f"  ❌ {category}: No CSV file found in {dir_name}")
            print(f"      Looked for: {[f.name for f in possible_files]}")
    
    if not csv_paths:
        print("\nNo CSV files found. Please run the model scripts first:")
        print("  python 1.linear_models.py")
        print("  python 2.kernel_models.py") 
        print("  python 3.tree_models.py")
        return
    
    # Hardcoded output directory
    output_dir = './task2_results'
    
    # Run analysis
    analyzer = Au20StructureAnalyzer(csv_paths, output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()