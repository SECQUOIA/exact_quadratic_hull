# Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs

This repository contains the code and computational results for the
research paper **"Exact Hull Reformulation for Quadratically Constrained
Generalized Disjunctive Programs"** by **Sergey Gusev** and **David E.
Bernal Neira** from the Davidson School of Chemical Engineering at
Purdue University.

The repository includes implementations of the proposed exact hull
reformulation for quadratically constrained generalized disjunctive
programs (QC-GDPs), as well as scripts and benchmark problems used in
the computational experiments reported in the paper.

------------------------------------------------------------------------

# Repository Structure

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
    │   ├── main/                    # K-means implementations and analysis scripts
    │   └── data/                    # Computational results
    ├── clay/                        # Constrained layout optimization benchmarks
    │   ├── clay.py                  # CLay batch execution script
    │   ├── analyze_clay_results.py  # Results analysis
    │   └── data/                    # Computational results
    ├── random_quadratic/            # Random quadratic GDP instances
    │   ├── random_quadratic/        # Main implementation and analysis scripts
    │   ├── data/                    # Generated instances and results
    │   └── tests/                   # Test files
    ├── requirements.txt             # Python dependencies
    └── LICENSE                      # MIT License

------------------------------------------------------------------------

# Benchmark Problems

The repository includes implementations and results for four types of
benchmark problems.

### 1. Random GDP Instances

Randomly generated quadratic GDP problems including both convex and
nonconvex instances.

### 2. CSTR Network Optimization

A superstructure optimization problem for a network of continuously
stirred tank reactors.

### 3. K-means Clustering

The classical unsupervised learning problem formulated as a generalized
disjunctive program.

### 4. Constrained Layout Optimization (CLay)

Two-dimensional facility layout optimization problems with quadratic
constraints.

------------------------------------------------------------------------

# Computational Environment

The computational experiments reported in the paper were conducted using
the following software versions:

-   **Python 3.12.9**
-   **Pyomo 6.9.2**
-   **GAMS 52.2.0**

The resulting MINLP formulations were solved through **GAMS** using the
following solvers (versions bundled with the corresponding GAMS
release):

-   **Gurobi 13.0**
-   **BARON 25.11.17**
-   **SCIP 9.2.4**

------------------------------------------------------------------------

# Environment Setup

## Step 1 --- Create Python Environment

Using conda (recommended):

    conda create -n exact_hull python=3.12 -y
    conda activate exact_hull

Verify installation:

    python --version

------------------------------------------------------------------------

## Step 2 --- Install Python Dependencies

From the repository root:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Step 3 --- Install GAMS (Skip if intstalled)

Download and install **GAMS 52.2.0** from:

https://www.gams.com/download/

After installation, add GAMS to your PATH.

Example for Linux:

    export GAMS_PATH="/opt/gams52.2_linux_x64_64_sfx"
    export PATH="$GAMS_PATH:$PATH"

Verify installation:

    gams

------------------------------------------------------------------------

## Step 4 --- Solver Licenses (Skip if intstalled)

GAMS requires a valid license.

The experiments also require licenses for the following solvers used
through GAMS:

-   Gurobi

These solvers are accessed through GAMS and correspond to the versions
bundled with the installed GAMS release.

Follow the licensing instructions provided by the respective solver
vendors.

------------------------------------------------------------------------

## Step 5 --- Install the Exact Hull Plugins

Clone the repository:

    git clone https://github.com/SECQUOIA/exact_quadratic_hull.git
    cd exact_quadratic_hull

Locate the Pyomo installation directory:

    PYOMO_PATH=$(python -c "import pyomo; print(pyomo.__path__[0])")
    echo "Pyomo path: $PYOMO_PATH"

Copy the reformulation plugins into the Pyomo plugin directory:

    cp addons/*.py "$PYOMO_PATH/gdp/plugins/"


------------------------------------------------------------------------

## Step 6 --- Verify Installation

Check Pyomo:

    python -c "import pyomo; print('Pyomo version:', pyomo.__version__)"

Verify plugin import:

    python -c "import pyomo.gdp.plugins.hull_exact; print('Exact hull plugin loaded successfully')"

------------------------------------------------------------------------

## Step 7 --- Run a Test Example

    cd random_quadratic/random_quadratic
    python batch_run.py --batch none

Results will appear in:

    random_quadratic/data/results.xlsx

------------------------------------------------------------------------

# Reproducing Experiments

    cd random_quadratic/random_quadratic

    python batch_run.py --batch psd
    python batch_run.py --batch nonconvex100

------------------------------------------------------------------------

# Citation

If you use this code or reference this work, please cite:

Gusev, S., & Bernal Neira, D. E. (2025).\
*Exact Hull Reformulation for Quadratically Constrained Generalized
Disjunctive Programs.*\
arXiv preprint arXiv:2508.16093.

https://arxiv.org/abs/2508.16093

------------------------------------------------------------------------

# License

This project is licensed under the MIT License.\
See the LICENSE file for details.
