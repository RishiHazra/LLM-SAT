#!/bin/bash
DIR='data/horn2/'
./scripts/gen_2horn_data.sh
python src/sat_solve.py $DIR
python plots/phase_transition.py phase_transition $DIR