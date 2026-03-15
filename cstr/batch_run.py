import os
import time
import json
from contextlib import redirect_stdout
from datetime import datetime

import pyomo.environ as pyo
from pyomo.opt.base.solvers import SolverFactory
from pyomo.core.expr.visitor import identify_variables
import gdp_reactor_gurobi
import pandas as pd
import shutil
import pyomo.gdp.plugins.hull_exact
import pyomo.gdp.plugins.hull_exact_conic
import pyomo.gdp.plugins.hull_exact_conic_no_sqrt_extra_var
import pyomo.gdp.plugins.hull_exact_conic_no_sqrt_no_extra_var
import pyomo.gdp.plugins.hull_exact_conic_original
import pyomo.gdp.plugins.hull_exact_conic_sqrt_extra_var
import pyomo.gdp.plugins.hull_exact_conic_sqrt_no_extra_var
import pyomo.gdp.plugins.hull_exact_conic_no_cholesky
import pyomo.gdp.plugins.hull_reduced_y
import pyomo.gdp.plugins.hull_exact_extra_var
import pyomo.gdp.plugins.hull_exact_extra_var_inequal

def _run_benchmark(num_reactors, mode, current_time, time_limit=3600, solvers=None):
    """
    Run the benchmark for the CSTR superstructure and return a list of summary dictionaries.

    Parameters
    ----------
    num_reactors : int
        Number of reactors in the CSTR superstructure.
    mode : str
        Mode of the CSTR superstructure.
    current_time : str
        Timestamp string used to name the results folder.
    time_limit : int, optional
        Time limit for the solver (default is 3600 seconds).
    solvers : list of str, optional
        List of solvers to run. Defaults to ["gurobi", "baron", "scip"].

    Returns
    -------
    summaries : list of dict
        A list of summary dictionaries for each solver run.
    """
    if solvers is None:
        solvers = ["gurobi", "baron", "scip"]
    summaries = []
    # Build the model.
    m = gdp_reactor_gurobi.build_model(NT=num_reactors, mode=mode)
    strategies = ["no_reformulation"]
    if mode == "original":
        strategies = ["gdp.bigm","gdp.hull", "gdp.hull_exact", "gdp.binary_multiplication"]
    
    # Clone the model to ensure a fresh solve.
    model_for_cloning = m.clone()
    results_dir_parent = os.path.join(
        os.getcwd(),
        "cstr",
        "benchmark_result",
        f"{num_reactors}_reactors",
        f"{mode} mode",
        current_time,
    )


    for strategy in strategies:
        m = model_for_cloning.clone()
        pyo.TransformationFactory("core.logical_to_linear").apply_to(m)
        pyo.TransformationFactory(strategy).apply_to(m)
        
        results_dir = os.path.join(results_dir_parent, strategy)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        for main_solver in ["gams"]:  # Add "gurobi" if desired.
            for solver in solvers:
                try:
                    # Start timing the solver.
                    start = time.time()
                    log_filename = (
                        f"{results_dir}/gams_{strategy}_{main_solver}_{solver}.log"
                        if main_solver == "gams"
                        else f"{results_dir}/gurobi_{strategy}_{main_solver}.log"
                    )
                    with open(log_filename, "w") as f:
                        with redirect_stdout(f):

                            if main_solver == "gurobi":
                                opt = SolverFactory(main_solver)
                                opt.options["NonConvex"] = 2
                                opt.options["TimeLimit"] = time_limit
                                opt.options["Threads"] = 1
                                opt.options["MIPGap"] = 1e-6
                                opt.options["MIPGapAbs"] = 0
                                res = opt.solve(m, tee=True)
                            elif main_solver == "gams":
                                TOLS = {
                                    "rel_gap": 1e-6,
                                    "abs_gap": 1e-10,
                                    "feas": 1e-6,
                                    "opt": 1e-6,
                                    "int": 1e-5,
                                }
                                if solver == "gurobi":
                                    options_gams = (
                                        '$onecho > gurobi.opt',
                                        'NonConvex 2',
                                        'Threads 1',
                                        f'MIPGap {TOLS["rel_gap"]}',
                                        f'MIPGapAbs {TOLS["abs_gap"]}',
                                        f'FeasibilityTol {TOLS["feas"]}',
                                        f'OptimalityTol {TOLS["opt"]}',
                                        f'IntFeasTol {TOLS["int"]}',
                                        '$offecho',
                                        'GAMS_MODEL.optfile=1;'
                                    )
                                elif solver == "baron":
                                    options_gams = (
                                        '$onecho > baron.opt',
                                        'Threads 1',
                                        f'EpsR {TOLS["rel_gap"]}',
                                        f'EpsA {TOLS["abs_gap"]}',
                                        f'AbsConFeasTol {TOLS["feas"]}',
                                        'RelConFeasTol 0',
                                        f'AbsIntFeasTol {TOLS["int"]}',
                                        'RelIntFeasTol 0',
                                        '$offecho',
                                        'GAMS_MODEL.optfile=1;'
                                    )
                                elif solver == "scip":
                                    options_gams = (
                                        '$onecho > scip.opt',
                                        f'limits/time = {time_limit}',
                                        'parallel/maxnthreads = 1',
                                        f'limits/gap = {TOLS["rel_gap"]}',
                                        f'limits/absgap = {TOLS["abs_gap"]}',
                                        f'numerics/feastol = {TOLS["feas"]}',
                                        f'numerics/dualfeastol = {TOLS["opt"]}',
                                        f'numerics/sumepsilon = {TOLS["feas"]}',
                                        'display/verblevel = 4',
                                        '$offecho',
                                        'GAMS_MODEL.optfile=1;'
                                    )
                                opt = pyo.SolverFactory("gams")
                                res = opt.solve(
                                    m,
                                    solver=solver,
                                    tee=True,
                                    keepfiles=True,
                                    tmpdir=results_dir,
                                    symbolic_solver_labels=True,
                                    add_options=[
                                        f"option reslim={time_limit};",
                                        "option threads=1;",
                                        f"option optcr={TOLS['rel_gap']};",
                                        f"option optca={TOLS['abs_gap']};",
                                        *options_gams,
                                    ],
                                )
                    end = time.time()
                    duration = end - start  # User solve time

                    # Annotate the result with some run info.
                    res.problem.name = m.name
                    res.problem.num_reactors = num_reactors
                    res.solver.solution_strategy = strategy
                    res.solver.subsolver = main_solver
                    res.solver.solver_gams = solver if main_solver == "gams" else None

                    # Extract model information.
                    if mode == "original":
                        # For original mode, extract from disjunct indicator variables.
                        activated_reactors = [n for n in m.N if m.YP_is_cstr[n].indicator_var()]
                        activated_recycle_flows = [n for n in m.N if m.YR_is_recycle[n].indicator_var()]
                        recycle_outlet_flow_rate = [m.QR() if m.YR_is_recycle[n].indicator_var() else 0 for n in m.N]
                        flow_A = [m.F['A', n]() for n in m.N if m.YP_is_cstr[n].indicator_var()]
                        flow_B = [m.F['B', n]() for n in m.N if m.YP_is_cstr[n].indicator_var()]
                        outlet_flow_rate = [m.Q[n]() for n in m.N if m.YP_is_cstr[n].indicator_var()]
                        reaction_rate_A = [m.rate['A', n]() for n in m.N if m.YP_is_cstr[n].indicator_var()]
                        volume = [m.c[n]() for n in m.N if m.YP_is_cstr[n].indicator_var()]
                    else:
                        # For hull_quadratic and naive_multiplication modes, use binary variables directly.
                        activated_reactors = [n for n in m.N if m.YP_is_cstr[n].value == 1]
                        activated_recycle_flows = [n for n in m.N if m.YR_is_recycle[n].value == 1]
                        recycle_outlet_flow_rate = [m.QR() if m.YR_is_recycle[n].value == 1 else 0 for n in m.N]
                        flow_A = [m.F['A', n]() for n in m.N if m.YP_is_cstr[n].value == 1]
                        flow_B = [m.F['B', n]() for n in m.N if m.YP_is_cstr[n].value == 1]
                        outlet_flow_rate = [m.Q[n]() for n in m.N if m.YP_is_cstr[n].value == 1]
                        reaction_rate_A = [m.rate['A', n]() for n in m.N if m.YP_is_cstr[n].value == 1]
                        volume = [m.c[n]() for n in m.N if m.YP_is_cstr[n].value == 1]


                    # Count number of constraints.
                    num_constraints = sum(1 for _ in m.component_data_objects(pyo.Constraint))
                    # Count variables.
                    num_variables = sum(1 for _ in m.component_data_objects(pyo.Var))
                    # Count binary, integer, and continuous variables.
                    num_binary = sum(1 for var in m.component_data_objects(pyo.Var) if var.domain == pyo.Binary)
                    num_integer = sum(1 for var in m.component_data_objects(pyo.Var)
                                      if var.domain in (pyo.Integers, pyo.NonNegativeIntegers))
                    num_continuous = sum(1 for var in m.component_data_objects(pyo.Var)
                                         if var.domain in (pyo.Reals, pyo.NonNegativeReals))
                    # Count "nonzeros": for each constraint, count distinct variable names in its body.
                    nonzeros = 0
                    for c in m.component_data_objects(pyo.Constraint, descend_into=True):
                        if c.body is not None:
                            # Use variable names (strings) to avoid unhashable type errors.
                            nonzeros += len({v.name for v in identify_variables(c.body)})
                    
                    # Build summary dictionary.
                    summary = {
                        "timestamp": current_time,
                        "model_name": m.name,
                        "strategy": strategy,
                        "main_solver": main_solver,
                        "solver": solver,
                        "mode": mode,
                        "num_reactors": num_reactors,
                        "solve_time_sec": duration,
                        "objective_value": pyo.value(m.obj),
                        "num_constraints": num_constraints,
                        "num_variables": num_variables,
                        "num_binary_variables": num_binary,
                        "num_integer_variables": num_integer,
                        "num_continuous_variables": num_continuous,
                        "num_nonzeros": nonzeros,
                        "activated_reactors": activated_reactors,
                        "activated_recycle_flows": activated_recycle_flows,
                        "recycle_outlet_flow_rate": recycle_outlet_flow_rate,
                        "flow_A": flow_A,
                        "flow_B": flow_B,
                        "outlet_flow_rate": outlet_flow_rate,
                        "reaction_rate_A": reaction_rate_A,
                        "volume": volume,
                    }
                    summaries.append(summary)

                    # Save extracted variable values to a JSON file.
                    os.makedirs(os.path.join(results_dir, "variables"), exist_ok=True)
                    variables_filename = (
                        f"{results_dir}/variables/gams_{strategy}_{main_solver}_{solver}.json"
                        if main_solver == "gams"
                        else f"{results_dir}/variables/gurobi_{strategy}_{main_solver}.json"
                    )
                    with open(variables_filename, "w") as f:
                        json.dump(
                            {
                                "objective_value": pyo.value(m.obj),
                                "activated_reactors": activated_reactors,
                                "activated_recycle_flows": activated_recycle_flows,
                                "recycle_outlet_flow_rate": recycle_outlet_flow_rate,
                                "flow_A": flow_A,
                                "flow_B": flow_B,
                                "outlet_flow_rate": outlet_flow_rate,
                                "reaction_rate_A": reaction_rate_A,
                                "volume": volume,
                                "hamming_distance": None,
                                "normalized_hamming_distance": None,
                            },
                            f,
                        )
                    results_filename = (
                        f"{results_dir}/gams_{strategy}_{main_solver}_{solver}.json"
                        if main_solver == "gams"
                        else f"{results_dir}/gurobi_{strategy}_{main_solver}.json"
                    )
                    with open(results_filename, "w") as f:
                        json.dump(res.json_repn(), f)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    error_filename = (
                        f"{results_dir}/gams_{strategy}_{main_solver}_{solver}.error"
                        if main_solver == "gams"
                        else f"{results_dir}/gurobi_{strategy}_{main_solver}.error"
                    )
                    with open(error_filename, "w") as f:
                        f.write(str(e))
    return summaries

if __name__ == "__main__":
    all_results = []
    NT = [i for i in range(5, 42)]  # Number of reactors to test.
    modes = ["original"]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_limit = 300
    all_solvers = ["gurobi", "baron", "scip"]

    # Track which solvers are done (all their strategies hit time limit)
    solvers_done = set()

    for num_reactors in NT:
        active_solvers = [s for s in all_solvers if s not in solvers_done]
        if not active_solvers:
            print(f"All solvers have reached time limit on all strategies. Stopping.")
            break

        print(f"\n--- {num_reactors} reactors: running solvers {active_solvers} ---")

        for mode in modes:
            summaries = _run_benchmark(
                num_reactors=num_reactors,
                mode=mode,
                current_time=current_time,
                time_limit=time_limit,
                solvers=active_solvers,
            )
            all_results.extend(summaries)
            print(f"Finished benchmark for {num_reactors} reactors with mode {mode}")

            # Determine the set of strategies used in this run
            unique_strategies = {s["strategy"] for s in summaries}

            # Check per-solver whether all strategies hit the time limit
            for solver in active_solvers:
                solver_summaries = [s for s in summaries if s["solver"] == solver]
                strategies_hit = {
                    s["strategy"]
                    for s in solver_summaries
                    if s["solve_time_sec"] >= time_limit - 1
                }
                if strategies_hit:
                    for st in strategies_hit:
                        print(f"  Time limit reached: solver={solver}, strategy='{st}', reactors={num_reactors}")
                if unique_strategies and strategies_hit == unique_strategies:
                    print(f"  => Solver '{solver}' hit time limit on ALL strategies at {num_reactors} reactors. Will skip it for higher reactor counts.")
                    solvers_done.add(solver)

    # Create an Excel summary of all benchmark runs.
    if all_results:
        output_excel = os.path.join(os.getcwd(), "cstr", "benchmark_summary.xlsx")
        new_df = pd.DataFrame(all_results)
        
        # Check if file already exists and append if it does
        if os.path.exists(output_excel):
            try:
                existing_df = pd.read_excel(output_excel)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(output_excel, index=False)
                print(f"Added new results to existing Excel summary at {output_excel}")
            except Exception as e:
                print(f"Error reading existing file, creating new file: {e}")
                new_df.to_excel(output_excel, index=False)
                print(f"New Excel summary saved to {output_excel}")
        else:
            new_df.to_excel(output_excel, index=False)
            print(f"Excel summary saved to {output_excel}")
    else:
        print("No benchmark results to summarize.")
