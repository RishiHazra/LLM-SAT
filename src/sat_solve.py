import os
import PyMiniSolvers.minisolvers as minisolvers
import pandas as pd
from time import time
from typing import List

from data_loader import data_loader
from data_loader import data_loader_from_chunks

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str, help="Dataset folder")

    args = parser.parse_args()

    dataset = data_loader_from_chunks(args.dataset_folder)
    n_formulas = len(dataset)
    solved = dict()
    solved["id"] = []
    solved["n"] = []
    solved["alpha"] = []
    solved["is_sat"] = []
    solved["time"] = []

    # Solve
    for id, n, m, formula, _, _ in dataset:
        solver = minisolvers.MinisatSolver()
        for _ in range(n):
            solver.new_var(dvar=True)
        for clause in formula:
            solver.add_clause(clause)

        print("Solving formula %s out of %s.." % (id, n_formulas))

        t0 = time()
        is_sat = solver.solve()
        solving_time = time() - t0

        solved["id"].append(id)
        solved["n"].append(n)
        solved["alpha"].append(m / n)
        solved["is_sat"].append(is_sat)
        solved["time"].append(solving_time)

    # Save
    df = pd.DataFrame.from_dict(solved)
    outfile = os.path.join(args.dataset_folder, "sat_solved.csv")
    df.to_csv(outfile, index=False)
