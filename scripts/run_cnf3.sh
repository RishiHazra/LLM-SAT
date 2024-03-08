#!/bin/bash
DIR='data/cnf3-small/'
./scripts/gen_cnf_data.sh
python src/sat_solve.py $DIR
python plots/phase_transition.py phase_transition $DIR