#!/bin/bash
DIR='data/cnf3-ra/'
./scripts/gen_cnf_rational_alpha.sh
python src/sat_solve.py $DIR
python plots/phase_transition.py phase_transition $DIR