#!/bin/bash

# dictionary of values of ns as key and alpha_inc as value
ns=("3" "4" "5" "6" "7" "8" "9" "10")
alpha_inc=("1" "0.25" "0.2" "0.5" "1" "0.125" "1" "0.1")

# loop over the ns_alpha_inc dictionary
DIR="data/cnf2/"
K=2
N=100
SEED=42
ALPHA_MIN=1
ALPHA_MAX=10
for index in ${!ns[*]}; do
  ALPHA_INC=${alpha_inc[$index]}
  NS=${ns[$index]}

  python src/data_maker.py cnf $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs
  # rename dataset_3sat_int.pkl to dataset-ns.pkl
  mv $DIR/dataset.pkl $DIR/dataset-$NS.pkl
done

# call merge function from data_maker.py on the directory DIR and then remove the formulas with int alphas
python src/data_maker.py merge_cnf $DIR $SEED  # SEED is not used but is required from the script (mayeb I'll fix it later)
mv $DIR/merged_dataset.pkl $DIR/dataset.pkl

# print datasets info in a file
python src/data_loader.py $DIR/dataset.pkl > $DIR/dataset_info.txt