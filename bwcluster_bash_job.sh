#!/bin/bash
#MSUB -l nodes=1:ppn=16:gpu,walltime=2:00:00:00 
#MSUB -N test
#MSUB -v PYTHON=devel/python_intel/3.6


module load $PYTHON
source activate tf2-gpu

cd hao/DE_HD_test/
python 02_keras_models_single.py
