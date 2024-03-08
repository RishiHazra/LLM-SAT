#!/bin/bash
DIR='data/cnf3-toy/'
K=3
N=100
SEED=42
ALPHA_MIN=2
ALPHA_MAX=9
ALPHA_INC=0.2
NS="5,10"

# Generate Data
python src/data_maker.py cnf $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs

# SAT-solve
python src/sat_solve.py $DIR

# Plot
python plots/phase_transition.py --extension ".png" phase_transition $DIR