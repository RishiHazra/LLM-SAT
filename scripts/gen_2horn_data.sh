#!/bin/bash
DIR='data/horn2/'
K=2
N=300
SEED=42
ALPHA_MIN=0
ALPHA_MAX=12
ALPHA_INC=1
NS="3,4,5,6,7,8,9,10"

python src/data_maker.py horn $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs


# print datasets info in a file
python src/data_loader.py $DIR/dataset.pkl > $DIR/dataset_info.txt