# Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs

This repository contains the code and computational results for the research paper **"Exact Hull Reformulation for Quadratically Constrained Generalized Disjunctive Programs"** by Sergey Gusev and David E. Bernal Neira from the Davidson School of Chemical Engineering at Purdue University.


## Repository Structure

```
exact_quadratic_hull/
├── pyomo_addon/
│   └── hull_exact.py            # Exact hull reformulation implementation
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

## Installation and Setup



### Software Versions Used

The computational experiments in this research were conducted using the following software versions:

- Python 3.12.9
- Pyomo 6.9.2
- GAMS 49.1.0
- Gurobi 12.0 
- BARON 25.2.1
- SCIP 9.2.0

To run the code in this repository, you need to install the custom Pyomo plugin and modify the GAMS writer. Follow these steps:

### Step 1: Install the Exact Hull Plugin

Copy the exact hull reformulation plugin ([`pyomo_addon/hull_exact.py`](pyomo_addon/hull_exact.py)) to your Pyomo installation:

```bash
# Find your Pyomo installation directory
python -c "import pyomo; print(pyomo.__path__[0])"

# Copy the plugin to the GDP plugins directory
cp pyomo_addon/hull_exact.py <PYOMO_PATH>/gdp/plugins/
```

Check that plugin is placed in the `pyomo/gdp/plugins/` directory of your Pyomo installation. Copy it manually if method above failed.

### Step 2: Modify the GAMS Writer (if using GAMS)

You need to modify the `gams_writer.py` file in your Pyomo installation:

1. The GAMS writer file is located at:
   ```
   <PYOMO_PATH>/pyomo/repn/plugins/gams_writer.py
   ```

2. Edit the file and find the line containing GUROBI solver capabilities (around line 1046):
   ```python
   'GUROBI': {'LP', 'NLP', 'MIP', 'MINLP', 'RMIP', 'QCP', 'MIQCP', 'RMIQCP'},
   ```

3. Ensure it matches exactly the line above (add 'NLP' if missing).



## Citation

If you use this code or reference this work, please cite:



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

