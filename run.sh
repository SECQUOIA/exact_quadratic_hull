#!/usr/bin/env bash

set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

jobs="${JOBS:-40}"
results_root="${RESULTS_ROOT:-results/full-gams54}"
conda_environment="${CONDA_ENVIRONMENT:-exact-quadratic-hull}"
gams_module="${GAMS_MODULE:-gams/54.3.1}"

if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer; received: $jobs" >&2
    exit 2
fi

# Lmod usually exports the module function to child Bash processes. Source the
# standard initialization file as a fallback for non-interactive shells.
if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module load "$gams_module"
fi

if [[ -z "${GRB_LICENSE_FILE:-}" && -r "$HOME/gurobi.lic" ]]; then
    export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$conda_environment" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is unavailable; activate $conda_environment before running this script." >&2
        exit 127
    fi
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$conda_environment"
fi

if ! command -v exact-hull >/dev/null 2>&1; then
    echo "exact-hull is unavailable in Conda environment $conda_environment." >&2
    exit 127
fi

doctor_output="$(exact-hull doctor)"
printf '%s\n' "$doctor_output"
for required in \
    "gams GDX results: available" \
    "gams/gurobi: available" \
    "gams/scip: available"
do
    if ! grep -Fq "$required" <<<"$doctor_output"; then
        echo "Preflight failed: expected '$required'." >&2
        exit 1
    fi
done

configs=(
    configs/random_psd.toml
    configs/random_psd_scip_convex.toml
    configs/random_psd_gurobi_auto.toml
    configs/random_psd_conic.toml
    configs/random_nonconvex.toml
    configs/eps_relaxation.toml
    configs/kmeans.toml
    configs/cstr.toml
    configs/clay.toml
)

mkdir -p "$results_root"
printf '%s\n' "$$" > "$results_root/launcher.pid"
printf 'Starting full experiment campaign with %s workers in %s\n' "$jobs" "$results_root"

exec exact-hull run \
    "${configs[@]}" \
    --resume "$results_root" \
    --strict-env \
    --jobs "$jobs"
