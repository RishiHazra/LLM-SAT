#!/bin/bash
DIR='data/cnf3-small/'
K=3
N=1
SEED=42
ALPHA_MIN=2
ALPHA_MAX=10
ALPHA_INC=1
NS=$(seq -s, 59, 80)

python src/data_maker.py cnf $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs
python src/data_maker.py merge_cnf $DIR $SEED
python src/sat_solve.py $DIR