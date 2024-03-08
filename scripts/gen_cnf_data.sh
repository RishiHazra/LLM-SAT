#!/bin/bash
DIR='data/cnf3-small/'
K=3
N=300
SEED=42
ALPHA_MIN=2
ALPHA_MAX=12
ALPHA_INC=1
NS="3,4,5,6,7,8,9,10"

python src/data_maker.py cnf $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs