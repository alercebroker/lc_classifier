#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# taskset -c 0 python profile_hierarchical_features.py many-lengths hierarchical_features.log

python profile_hierarchical_features.py many-lengths hierarchical_features.log