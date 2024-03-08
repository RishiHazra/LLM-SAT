#!/bin/bash
DIR='data/cnf2/'
./scripts/gen_cnf2.sh
python src/sat_solve.py $DIR
python plots/phase_transition.py phase_transition $DIR