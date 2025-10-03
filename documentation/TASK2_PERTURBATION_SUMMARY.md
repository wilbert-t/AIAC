# Task2 Model Perturbation Analysis Summary

## Model Performance Under Structural Perturbations

**Baseline Performance**: The Task2 model started with a higher baseline prediction error of 2.84 eV across 10 test structures covering an energy range of -1557.21 to -1555.94 eV, indicating a broader and more challenging energy landscape.

**Perturbation Testing Protocol**: Comprehensive stress testing involved atomic displacements across varying atom counts (1-3 atoms) and perturbation strengths (weak=2, medium=5, strong=9), resulting in 90 total analyses to evaluate model stability under structural modifications.

**Robustness Results**: Task2 showed higher structural sensitivity with an overall mean sensitivity of 0.397 ± 0.261 eV/Å. The model exhibited significant sensitivity scaling: single-atom perturbations (0.255 eV/Å), two-atom perturbations (0.401 eV/Å), and three-atom perturbations (0.535 eV/Å).

**Performance Metrics**:
- **RMSE Range**: 4.28-4.98 eV showing substantial error variation
- **MAE Range**: 4.28-4.97 eV indicating lower overall accuracy
- **Sensitivity Scaling**: 61% increase from weak to strong perturbations
- **Top Performers**: Structures 804 and 351 demonstrated best stability (0.284-0.334 eV/Å sensitivity)

**Key Challenges**: Higher baseline errors, greater sensitivity to structural changes, and more variable performance under perturbations. While still functional, the model requires more careful handling of input structural quality and shows limitations for high-precision applications requiring structural robustness.