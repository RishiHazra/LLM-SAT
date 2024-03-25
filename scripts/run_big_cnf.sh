DIR='data/cnf3/'
K=3
N=300
SEED=42
ALPHA_MIN=1
ALPHA_MAX=12
ALPHA_INC=0.1
NS="10,20,30,50,100,200"

python src/data_maker.py cnf $ALPHA_MIN $ALPHA_MAX -N $N -k $K --alpha_inc $ALPHA_INC --ns $NS $DIR $SEED #--dimacs
python src/sat_solve.py $DIR
python plots/phase_transition.py phase_transition $DIR