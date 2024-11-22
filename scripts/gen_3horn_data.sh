#!/bin/bash

# dictionary of values of ns as key and alpha_inc as value
# ns=("3" "5" "10" "1000" "10000")
# alpha_inc=("1" "0.2" "0.1" "0.1" "0.1")

# ns=("3" "5" "10")
# alpha_inc=("1" "0.4" "0.3")

# # loop over the ns_alpha_inc dictionary
# DIR="data/horn3-high-alpha/"
# K=3
# N=300
# SEED=42
# ALPHA_MIN=10
# ALPHA_MAX=20
# for index in ${!ns[*]}; do
#   ALPHA_INC=${alpha_inc[$index]}
#   NS=${ns[$index]}

#   python src/data_maker.py horn $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs
#   # rename dataset.pkl to dataset-ns.pkl
#   mv $DIR/dataset.pkl $DIR/dataset-$NS.pkl
# done

# # call merge function from data_maker.py on the directory DIR
# python src/data_maker.py merge_cnf $DIR $SEED  # SEED is not used but is required from the script (mayeb I'll fix it later)
# mv $DIR/merged_dataset.pkl $DIR/dataset.pkl

#!/bin/bash
DIR='data/horn3/'
K=3
N=300
SEED=42
ALPHA_MIN=0
ALPHA_MAX=12
ALPHA_INC=1
NS="3,4,5,6,7,8,9,10"

python src/data_maker.py horn $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs


# print datasets info in a file
python src/data_loader.py $DIR/dataset.pkl > $DIR/dataset_info.txt