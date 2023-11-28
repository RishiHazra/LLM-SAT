import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from verify_solution import verify_solution
from data_analysis import plot_comparison, plot_distribution, plot_dicts, plot_bar_comparison
import Levenshtein as lev
from collections import defaultdict


def find_best_match(target, string_list):
    # Initialize the best match with the first string in the list and maximum distance
    best_match = string_list[0]
    min_distance = lev.distance(target, best_match)

    # Iterate over the list to find the string with the smallest Levenshtein distance to the target
    for s in string_list[1:]:
        distance = lev.distance(target, s)
        if distance < min_distance:
            min_distance = distance
            best_match = s

    return best_match

def parse_generated_output(raw_output: str) -> Tuple[List[str], List[str]]:
    # Regular expression to find the lists
    pattern1 = r"```python\s+orderable\s*=\s*(\[[^\]]*\])\s*(,|\n)\s*not_orderable\s*=\s*(\[[^\]]*\])\s*```"
    match1 = re.search(pattern1, raw_output)

    pattern2 = r"orderable\s*=\s*(\[[^\]]*\])\s*(,|\n)\s*not_orderable\s*=\s*(\[[^\]]*\])\s*"
    match2 = re.search(pattern2, raw_output)

    # Extracting the lists if found
    if match1:
        try:
            orderable_list = eval(match1.group(1))
        except NameError:
            orderable_list = match1.group(1).strip('[]').split(',')
        except SyntaxError:
            orderable_list = match1.group(1).strip('[]').split(',')
            orderable_list = [item.strip().strip("''") for item in orderable_list]
        try:
            not_orderable_list = eval(match1.group(3))
        except NameError:
            not_orderable_list = match1.group(3).strip('[]').split(',')
        except SyntaxError:
            not_orderable_list = match1.group(3).strip('[]').split(',')
            not_orderable_list = [item.strip().strip("''") for item in not_orderable_list]
    elif match2:
        final_lists = re.findall(pattern2, raw_output)[-1]
        orderable_list, not_orderable_list = final_lists[0], final_lists[-1]
        if orderable_list == '[]':
            orderable_list = []
        else:
            orderable_list = orderable_list.strip('[]').split(',')
        if not_orderable_list == '[]':
            not_orderable_list = []
        else:
            not_orderable_list = not_orderable_list.strip('[]').split(',')
    # TODO: also check for cases where GPT-4 gave up
    else:
        orderable_list = not_orderable_list = []

    # postprocess
    orderable_list = [item.strip().strip("''").strip('""').lower() for item in orderable_list]
    not_orderable_list = [item.strip().strip("''").strip('""').lower() for item in not_orderable_list]
    return orderable_list, not_orderable_list


if __name__ == "__main__":
    model_name = 'llama-2-70b'  # gpt-3.5, gpt-4, llama-2-70b
    file_lines = open(f'../out_data_{model_name}/data_log.log', 'r').readlines()
    correct = 0
    dataframe_dict = {'num_variables':[], 'num_clauses': [], 'alpha':[], 'is_sat':[],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': []}

    for id, line in enumerate(file_lines):
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

        # if not sample_dict['is_sat']:   # if unsat
        #     continue

        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])

        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)

        try:
            orderable, not_orderable = \
                parse_generated_output(raw_output=sample_dict['gpt_out'])
        except AttributeError:
            orderable, not_orderable = [], []

        if not sample_dict['is_sat']:  # if unsat
            if orderable == [] and not_orderable == []:  # gpt predicts unsat
                dataframe_dict['correct'].append(True)
            else:
                dataframe_dict['correct'].append(False)
        else:  # if sat, verify solution
            if orderable == [] and not_orderable == []:  # gpt predicts unsat
                dataframe_dict['correct'].append(False)
            else:
                assignment = []
                for item in orderable:
                    item = find_best_match(item, sample_dict['menu_items'])
                    assignment.append(sample_dict['menu_items'].index(item)+1)
                for item in not_orderable:
                    item = find_best_match(item, sample_dict['menu_items'])
                    assignment.append(-sample_dict['menu_items'].index(item)-1)
                verified = \
                    verify_solution(num_vars=sample_dict['num_vars'],
                                    formula=sample_dict['formula'],
                                    assignment=assignment)
                if verified:
                    dataframe_dict['correct'].append(True)
                else:
                    dataframe_dict['correct'].append(False)

    # Creating the DataFrame
    df = pd.DataFrame(dataframe_dict)
    # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
    accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
    lambda x: np.mean(x['correct'])
).reset_index(name='accuracy')


    # Plotting
    plt.figure(figsize=(10, 6))
    for num_vars in accuracy_df['num_variables'].unique():
        # if num_vars in [4,6,10]:
            subset = accuracy_df[accuracy_df['num_variables'] == num_vars]
            plt.plot(subset['alpha'], subset['accuracy'], marker='o', label=f'Variables: {num_vars}')

    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.title(f'{model_name} accuracy vs alpha')
    plt.grid(True)
    plt.show()

    accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_df_simple['alpha'], accuracy_df_simple['accuracy'], marker='o')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.title(f'{model_name} accuracy vs alpha')
    plt.grid(True)
    plt.show()

    _3d_df_simple = df.groupby('num_variables')['correct'].mean().reset_index(name='accuracy')
    plt.figure(figsize=(10, 6))
    plt.plot(_3d_df_simple['num_variables'], _3d_df_simple['accuracy'], marker='o')
    plt.xlabel('# variables')
    plt.ylabel('accuracy')
    plt.title(f'{model_name} accuracy vs # variables')
    plt.grid(True)
    plt.show()

    _3d_df_simple = df.groupby('num_clauses')['correct'].mean().reset_index(name='accuracy')
    plt.figure(figsize=(10, 6))
    plt.plot(_3d_df_simple['num_clauses'], _3d_df_simple['accuracy'], marker='o')
    plt.xlabel('# clauses')
    plt.ylabel('accuracy')
    plt.title(f'{model_name} accuracy vs # clauses')
    plt.grid(True)
    plt.show()
    # Plot distributions
    # plot_distribution(num_vars_list, 'Distribution of num_vars', 'num_vars', 'Frequency')
    # plot_distribution(num_clauses_list, 'Distribution of num_clauses', 'num_clauses', 'Frequency')
    # # plot_distribution(is_sat_list, 'Distribution of is_sat', 'is_sat', 'Frequency')
    # plot_distribution(num_prompt_tokens_list, 'Distribution of Prompt Tokens', 'num_prompt_tokens', 'Frequency')
    # plot_distribution(num_completion_tokens_list, 'Distribution of Completion Tokens', 'num_completion_tokens', 'Frequency')
    # #
    # # Plot comparisons
    # plot_comparison(num_vars_list, num_prompt_tokens_list, 'num_vars', 'num_prompt_tokens', 'Comparison of num_vars and num_prompt_tokens')
    # plot_comparison(num_clauses_list, num_completion_tokens_list, 'num_classes', 'num_completion_tokens', 'Comparison of num_clauses and num_completion_tokens')
    # plot_comparison(alpha_list, num_prompt_tokens_list, 'alpha', 'num_prompt_tokens',
    #                 'Comparison of alpha and num_prompt_tokens')
    # plot_comparison(alpha_list, num_completion_tokens_list, 'alpha', 'num_completion_tokens',
    #                 'Comparison of alpha and num_completion_tokens')
    # plot_bar_comparison(alpha_list, is_sat_list, 'alpha', 'num is_sat', '# is_sat')
    # # plot_dicts(var_correct, 'num vars', 'accuracy', 'accuracy vs. num vars')
    # # plot_dicts(clause_correct, 'num clauses', 'correct', 'accuracy vs. num clauses')
    # plot_dicts(alpha_correct, 'alpha', 'accuracy', 'accuracy vs. alpha')


