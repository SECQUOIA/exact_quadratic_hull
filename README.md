# Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs

This repository contains the code and computational results for the research paper **"Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs"** by Sergey Gusev and David E. Bernal Neira from the Davidson School of Chemical Engineering at Purdue University.


## Repository Structure

```
exact_quadratic_hull/
├── addons/
│   ├── hull_exact.py            # Exact hull reformulation implementation
│   ├── hull_reduced_y.py        # Reduced hull reformulation implementation
│   └── gams_writer.py           # Modified GAMS writer for Pyomo 6.9.2
├── cstr/                        # CSTR network optimization benchmarks
│   ├── gdp_reactor.py           # CSTR GDP model implementation
│   ├── batch_run.py             # Batch execution script
│   └── data/                    # Computational results
├── k_means/                     # K-means clustering benchmarks
│   ├── main/                    # K-means implementations and analysys python files
│   └── data/                    # Computational results
├── clay/                        # Constrained layout optimization benchmarks
│   ├── clay.py                  # Clay batch execution script
│   ├── analyze_clay_results.py  # Results analysis
│   └── data/                    # Computational results
├── random_quadratic/            # Random quadratic GDP instances
│   ├── random_quadratic/        # Main implementation and analysys python files
│   ├── data/                    # Generated instances and results
│   └── tests/                   # Test files
└── LICENSE                      # MIT License
```

## Benchmark Problems

The repository includes implementations and results for four types of benchmark problems:

1. **Random GDP Instances**: Randomly generated quadratic GDP problems (both convex and non-convex)
2. **CSTR Network Optimization**: Continuously Stirred Tank Reactor superstructure optimization
3. **K-means Clustering**: Classic unsupervised learning problem formulated as GDP
4. **Constrained Layout Optimization (CLay)**: 2D facility layout problems with quadratic constraints

## Environment Setup from Scratch

This section provides complete instructions for setting up the computational environment from scratch.

### Software Versions Used

The computational experiments in this research were conducted using the following software versions:

- Python 3.12.9
- Pyomo 6.9.2
- GAMS 49.1.0
- Gurobi 12.0 
- BARON 25.2.1
- SCIP 9.2.0

### Step 1: Create and Activate Conda Environment

```bash
# Create conda environment with Python 3.12
conda create -n exact_hull python=3.12 -y

# Activate the environment
conda activate exact_hull

# Verify Python version
python --version
```

### Step 2: Install Required Packages via Conda and Pip

```bash
# Install core packages via conda (faster and more reliable)
conda install -c conda-forge pyomo=6.9.2 -y
conda install -c conda-forge numpy scipy matplotlib pandas -y



# Install additional packages 
conda install -c conda-forge pandas openpyxl -y
conda install -c conda-forge dill
conda install -c conda-forge openpyxl

```

### Step 3: Install GAMS

**Download and install GAMS 49.1.0:**

1. Visit [GAMS Download Page](https://www.gams.com/download/)
2. Download GAMS 49.1.0 for your operating system
3. Install GAMS following the installation wizard
4. **Important**: Note the GAMS installation directory (typically `/opt/gams49.1_linux_x64_64_sfx` on Linux)

**Set up GAMS environment:**

```bash
# Add GAMS to PATH (adjust path to your GAMS installation)
export GAMS_PATH="/opt/gams49.1_linux_x64_64_sfx"
export PATH="$GAMS_PATH:$PATH"

# Verify GAMS installation
gams 
```

### Step 4: Solver Licenses

**GAMS License (Required):**
- GAMS requires a license to run
- Set up license as instructed during GAMS installation

**Gurobi License (Required for Gurobi solver):**
- Download from https://www.gurobi.com/downloads/
- Academic licenses are available for free
- Set up license file as instructed by Gurobi
- **Note**: You need this license to use Gurobi solver through GAMS


### Step 5: Install the Exact Hull Plugin

```bash
# Clone or download this repository
git clone https://github.com/SECQUOIA/exact_quadratic_hull.git
cd exact_quadratic_hull

# Find your Pyomo installation directory
PYOMO_PATH=$(python -c "import pyomo; print(pyomo.__path__[0])")
echo "Pyomo path: $PYOMO_PATH"

# Copy the exact hull plugins to Pyomo
cp addons/hull_exact.py "$PYOMO_PATH/gdp/plugins/"
cp addons/hull_reduced_y.py "$PYOMO_PATH/gdp/plugins/"

# Verify the plugins are installed
ls "$PYOMO_PATH/gdp/plugins/hull_exact.py"
ls "$PYOMO_PATH/gdp/plugins/hull_reduced_y.py"
```

### Step 6: Modify the GAMS Writer (Required for GAMS)

**Issue**: Pyomo 6.9.2 doesn't include the 'NLP' and 'MINLP' flag for Gurobi solver capabilities, which is required for MINLP problems. This was fixed in later Pyomo versions.

**Solution**: You can either manually edit the GAMS writer file or use the pre-modified version from this repository.

**Option 1: Use the pre-modified GAMS writer (Recommended)**

```bash
# Locate the GAMS writer file
GAMS_WRITER="$PYOMO_PATH/repn/plugins/gams_writer.py"
echo "GAMS writer location: $GAMS_WRITER"


# Copy the modified GAMS writer from this repository
cp addons/gams_writer.py "$GAMS_WRITER"
```

**Option 2: Manual edit**

```bash
# Locate the GAMS writer file
GAMS_WRITER="$PYOMO_PATH/repn/plugins/gams_writer.py"

# Open the file for editing
nano "$GAMS_WRITER"
# OR use your preferred editor:
# vim "$GAMS_WRITER"
# code "$GAMS_WRITER"  # VS Code
# gedit "$GAMS_WRITER"  # GNOME Text Editor

# Edit the file to ensure GUROBI capabilities include 'NLP' and 'MINLP'
# Find line ~1046 and ensure it contains:
# 'GUROBI': {'LP', 'NLP', 'MIP', 'MINLP', 'RMIP', 'QCP', 'MIQCP', 'RMIQCP'}

```

### Step 7: Verify Installation

```bash
# Make sure conda environment is activated
conda activate exact_hull

# Test Python environment
python -c "import pyomo; print('Pyomo version:', pyomo.__version__)"
python -c "import pyomo.gdp.plugins.hull_exact; print('Exact hull plugin loaded successfully')"

# Test solver availability and licensing
python -c "
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Test GAMS (required)
try:
    solver = SolverFactory('gams')
    print('GAMS available:', solver.available())
except:
    print('GAMS not available')

# Test GAMS subsolvers with simple problem
def test_solver(solver_name):
    try:
        # Create a simple test problem
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, 10))
        model.obj = pyo.Objective(expr=model.x**2, sense=pyo.minimize)
        
        # Test solver
        solver = SolverFactory('gams')
        solver.options['solver'] = solver_name
        
        if solver.available():
            result = solver.solve(model, tee=False)
            if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                print(f'GAMS-{solver_name}: Available and licensed ✓ (obj of test problem: {pyo.value(model.obj):.6f})')
            else:
                print(f'GAMS-{solver_name}: Available but license issue (status: {result.solver.termination_condition})')
        else:
            print(f'GAMS-{solver_name}: Not available')
    except Exception as e:
        print(f'GAMS-{solver_name}: Error - {str(e)[:50]}...')

# Test each subsolver
test_solver('gurobi')
test_solver('baron') 
test_solver('scip')
"
```

### Step 8: Run a Test Example

```bash
# Make sure conda environment is activated
conda activate exact_hull


# Test with random quadratic instances
cd random_quadratic/random_quadratic
python batch_run.py --batch none  # Test random instances
```

### Step 9: Replicate Results for Random Quadratics

To replicate the computational results for random quadratic instances presented in the paper, run the following commands:

```bash
# Make sure conda environment is activated
conda activate exact_hull

# Navigate to the random quadratic directory
cd random_quadratic/random_quadratic

# Run batch experiments for positive semidefinite (convex) instances
python batch_run.py --batch psd

# Run batch experiments for non-convex instances  
python batch_run.py --batch nonconvex100
```

These commands will generate comprehensive computational results for both convex (psd) and non-convex (non) random quadratic GDP instances, which can be used to reproduce the performance comparisons and analysis presented in the research paper.

**Congratulations!** If the test ran successfully, you have successfully set up the exact hull reformulation environment. You can find the logs of the test run in the `data/` folder, and a summary of test run results will be saved in `projects/env_test/exact_quadratic_hull/random_quadratic/data/results.xlsx`.

### Troubleshooting

**Common Issues:**

1. **GAMS not found**: Ensure GAMS is in your PATH and environment variables are set correctly
2. **Solver license issues**: Verify solver licenses are properly configured
3. **Plugin not loading**: Check that `hull_exact.py` is in the correct Pyomo plugins directory
4. **GAMS writer modification**: Ensure the GUROBI capabilities line includes 'NLP' and 'MINLP'



## Citation

If you use this code or reference this work, please cite:

Gusev, S., & Bernal Neira, D. E. (2025). Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs. arXiv preprint arXiv:2508.16093. https://arxiv.org/abs/2508.16093



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

