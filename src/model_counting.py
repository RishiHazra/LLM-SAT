# Import necessary libraries
import os
import ast
from data_maker import DataMaker as dm


def execute_cmd(cmd):
    """Execute the given command in the shell."""
    import subprocess
    import shlex

    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = process.communicate()
    return output, error


def model_counting(formula, n) -> int:
    formula_str = dm.to_dimacs(n, formula)

    # create tmp if not exists
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    # write formula to a temporary file
    filepath = "tmp/formula.cnf"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formula_str)

    d4_output, _ = execute_cmd(f"timeout 5m ./src/d4/d4 -mc {filepath}")
    model_count = int(d4_output.splitlines()[-1][1::])
    return model_count


def main():
    # this takes in input the output from the LLM execution
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Log file from LLM execution")
    args = parser.parse_args()
    log_file = args.log_file

    # retrieve directory from log_file
    log_directory = os.path.dirname(log_file)

    file_lines = open(log_file, "r").readlines()

    with open(f"{log_directory}/data_log_mc.log", "w", encoding="utf-8") as f:
        for id, line in enumerate(file_lines):
            print(f"Model counting formula {id+1}/{len(file_lines)}...")
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # Compute the model counting
            sample_dict["model_count"] = model_counting(
                sample_dict["formula"], sample_dict["num_vars"]
            )

            # write sample dict and model count to file
            f.write(f"Data Sample: {sample_dict}\n")


if __name__ == "__main__":
    main()
