import os
import pickle
import numpy as np
from data_maker import DataMakerSR as dm
from data_maker import pickle_load_from_chunks


def data_loader(dataset_path):
    dataset = None
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    return dataset


def data_loader_from_chunks(dataset_folder):
    return pickle_load_from_chunks(dataset_folder)


def data_stats(dataset):
    vars_per_formula = []
    clauses_per_formula = []
    alphas_per_n = {}
    tot_sat = 0

    for formulas_idx, n, m, formula, is_sat, _ in dataset:
        vars_per_formula.append(n)
        clauses_per_formula.append(m)

        # store alpha (m/n) per each n in alphas_per_n without duplicates
        if n not in alphas_per_n:
            alphas_per_n[n] = set()
        alphas_per_n[n].add(round(m / n, 3))

        if is_sat:
            tot_sat += 1

    print("Tot formulas: %s (%s are SAT)" % (len(dataset), tot_sat))
    # loop over alpha_per_n and print the number of alphas per n sorted by n, where alphas are also sorted
    print("\nAlphas per n:")
    for n in sorted(alphas_per_n.keys()):
        print("n=%s: %s" % (n, sorted(alphas_per_n[n])))

    vars_details_str = "\nAvg number of variables: %.2f (std %.2f)" % (
        sum(vars_per_formula) / len(vars_per_formula),
        np.std(vars_per_formula),
    )
    vars_details_str += "\nMax. number of variables: %d" % max(vars_per_formula)
    vars_details_str += "\nMin. number of variables: %d" % min(vars_per_formula)
    print(vars_details_str)

    clauses_details_str = "\nAvg number of clauses: %.2f (std %.2f)" % (
        sum(clauses_per_formula) / len(clauses_per_formula),
        np.std(clauses_per_formula),
    )
    clauses_details_str += "\nMax. number of clauses: %d" % max(clauses_per_formula)
    clauses_details_str += "\nMin. number of clauses: %d" % min(clauses_per_formula)
    print(clauses_details_str)

    alphas_per_formula = [a / b for a, b in zip(clauses_per_formula, vars_per_formula)]
    alphas_details_str = "\nAvg alpha: %.2f (std %.2f)" % (
        sum(alphas_per_formula) / len(alphas_per_formula),
        np.std(alphas_per_formula),
    )
    alphas_details_str += "\nMax. alpha: %d" % max(alphas_per_formula)
    alphas_details_str += "\nMin. alpha: %d" % min(alphas_per_formula)
    print(alphas_details_str)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Dataset path")

    args = parser.parse_args()

    dataset = data_loader(args.dataset_path)
    data_stats(dataset)

    # you can also generate the dimacs of a formula on the fly
    formulas_idx, n, m, formula, is_sat, _ = dataset[0]
    print("\nDIMACS:")
    print(dm.to_dimacs(n, formula))
