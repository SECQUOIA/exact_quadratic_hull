# Installation

Create the declared Conda environment from the repository root:

```console
conda env create -f environment.yml
conda activate exact-quadratic-hull
exact-hull doctor
```

The package pins Pyomo 6.10.0 because the transformations override its private `Hull_Reformulation._transform_constraint` method. The test suite has a canary for that method's signature.

## GAMS and solvers

Install GAMS using the vendor instructions and put its executable directory on `PATH`. Confirm with `gams`. Experiment runs use the GAMS solver interface and one of Gurobi, BARON, or SCIP. A GAMS license and any solver-specific licenses required by your GAMS distribution must cover all three solvers if you reproduce every campaign. The SCIP `convex` run is a SCIP option variant, not a transformation.

`exact-hull doctor` reports Python and Pyomo versions, all package transformation registrations, the GAMS executable, and whether Pyomo can use GDX result files. It warns when GDX support is unavailable because Pyomo's `dat` fallback crashes on the `OBJVAL NA` records commonly produced by timeouts. When GAMS is present, the command also attempts a tiny solve with Gurobi, BARON, and SCIP to expose missing solver installations or licenses. No GAMS installation is needed for unit tests or dry runs.
