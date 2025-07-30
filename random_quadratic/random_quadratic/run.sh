#!/bin/bash -l

module load gurobi/12.0
module load gams/49.1.0

export GRB_LICENSE_FILE=$HOME/gurobi.lic

python batch_run.py
