#!/usr/bin/env bash
# Historical Moab template: adapt resources to your cluster before submitting.
#MSUB -l nodes=1:ppn=16:gpu,walltime=2:00:00:00
#MSUB -N osm4eo-training

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -eq 0 ]; then
    echo "Pass training arguments, including --train, --validation and --output-model." >&2
    exit 2
fi

# Activate your environment before submission, or set OSM4EO_PYTHON explicitly.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
exec "${OSM4EO_PYTHON:-python}" 02_keras_models_single.py "$@"
