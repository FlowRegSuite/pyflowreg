# Synthetic Benchmarks for Motion Correction Methods

This directory contains benchmarks for comparing different motion correction methods on synthetic data with known ground truth displacement fields.

## Setup Instructions

### Prerequisites

The benchmarks require several Python packages for different motion correction methods. Due to the different installation requirements, we recommend using conda/mamba for the environment setup.

### Creating the Environment

1. **Create a new conda environment:**
```bash
conda create -n pyflowreg-bench python=3.9
conda activate pyflowreg-bench
```

2. **Install CaImAn (for NoRMCorre):**

**IMPORTANT**: The package named `caiman` on PyPI is NOT the correct package - it's a MicroPython build system. You need the CaImAn calcium imaging analysis package from Flatiron Institute.

Install the correct CaImAn package using conda/mamba:
```bash
# Using mamba (recommended, faster)
mamba install -c conda-forge caiman

# OR using conda
conda install -c conda-forge caiman
```

Alternatively, install from source:
```bash
git clone https://github.com/flatironinstitute/CaImAn.git
cd CaImAn
pip install .
cd ..
```

3. **Install other dependencies via pip:**
```bash
# Install the base pyflowreg package first (from project root)
cd ../..  # Navigate to pyflowreg root
pip install -e .

# Return to benchmarks directory
cd benchmarks/synth

# Install benchmark-specific requirements
pip install suite2p
pip install antspyx
pip install itk-elastix
pip install h5py
pip install tifffile
pip install numpy
pip install scipy
pip install pandas
pip install opencv-python
```

### Verifying Installation

Check that all packages are correctly installed:
```python
python -c "from caiman.motion_correction import MotionCorrect; print('CaImAn OK')"
python -c "import suite2p; print('Suite2p OK')"
python -c "import ants; print('ANTsPy OK')"
python -c "import itk; print('ITK-Elastix OK')"
```

## Running Benchmarks

### Basic Usage

Run the evaluation script:
```bash
python evaluation.py
```

### Options

- `--data`: Path to synthetic data file (downloads automatically if not provided)
- `--out`: Output directory for results (default: `out`)
- `--split`: Dataset split to use (`clean`, `noisy35db`, `noisy30db`)
- `--params`: Directory containing Elastix parameter files

### Example with specific dataset:
```bash
python evaluation.py --split noisy35db --out results_noisy35
```

## Methods Benchmarked

1. **PyFlowReg**: Variational optical flow method (this package)
2. **Suite2p**: Both rigid and non-rigid registration
3. **NoRMCorre** (via CaImAn): Non-rigid motion correction
4. **ANTsPy**: Multiple variants (SyN, ElasticSyN with different metrics)
5. **Elastix**: B-spline registration with various metrics

## Output

The benchmarks produce:
- `results_summary.csv`: Summary statistics for each method
- `results_detailed.csv`: Detailed results for each frame pair
- Individual `.h5` files containing displacement fields for each method
- LaTeX table in `results_table.tex`

## Metrics

- **EPE**: End-point error (average displacement error)
- **EPE95**: 95th percentile of displacement errors
- **Curl**: Mean absolute curl of the displacement field (smoothness measure)
- **Time**: Computation time in seconds

## Troubleshooting

### "No module named 'caiman'" Error
You have the wrong `caiman` package installed. Uninstall it and install the correct one:
```bash
pip uninstall caiman  # Remove wrong package
conda install -c conda-forge caiman  # Install correct package
```

### NoRMCorre tests skipped
If you see "Warning: CaImAn/NoRMCorre not available, skipping those tests", ensure CaImAn is properly installed via conda as described above.

### Memory Issues
Some methods (especially on large images) may require significant memory. Consider:
- Reducing image size for testing
- Running methods individually
- Increasing system swap space

## Notes on CaImAn/NoRMCorre

The NoRMCorre implementation uses CaImAn's motion correction module with parameters matched to the reference implementation:
- Grid size: 16×16 pixels
- Maximum shift: 25 pixels  
- Piecewise-rigid correction enabled
- Gaussian preprocessing with σ=(0.5, 0.5)
- Grayscale normalization applied

The implementation in `methods/normcorre.py` handles the conversion between CaImAn's shift representation and the standard displacement field format used by other methods.