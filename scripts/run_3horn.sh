#!/bin/bash
DIR='data/horn3/'
./scripts/gen_3horn_data.sh
python src/sat_solve.py $DIR
python plots/phase_transition.py phase_transition $DIR