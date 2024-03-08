#!/bin/bash
DIR='data/sr10-40/'
PAIRS=5000
SEED=42
MIN_VARS=10
MAX_VARS=40

python3 sr src/data_maker.py $DIR $PAIRS $MIN_VARS $MAX_VARS $SEED #--dimacs