import os.path
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Dict
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

def parse_generated_output_sat(raw_output: str) -> List:

    # Regex pattern to match the specific format with numbers and boolean values
    pattern = r"\n<output:\s*{\s*(?:(-?\d+): (True|False),?\s*)*}>"

    # Search for the pattern in the string
    match = re.search(pattern, raw_output)

    # If a match is found, process it
    if match:
        # Find all key-value pairs in the matched string
        kv_pairs = re.findall(r"(-?\d+): (True|False)", match.group(0))

        # Process each key-value pair
        result_list = []
        for key, value in kv_pairs:
            # Convert key to int, handling negative keys
            int_key = int(key)
            if int_key < 0:
                int_key = abs(int_key)
                value = "False" if value == "True" else "True"

            # Add the key-value pair to the result dictionary
            if value == "True":
                result_list.append(int_key)

        return result_list
    # If no match is found, return an empty dictionary
    return []


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


# Further refined function to correctly parse the CNF formula from LaTeX format
def parse_generated_output_translate(raw_output: str, item_to_number: Dict[str, int]) -> List[List]:
    # Extract the clauses from the LaTeX formatted string
    # Remove all LaTeX formatting and split by 'and' (\\land)
    raw_output = raw_output.replace('lnot', 'neg').replace('wedge', 'land').replace('vee','lor').replace('\\text{', '').replace('}', '').replace(r'\\', '')
    raw_output = raw_output.replace('\\\\\n&','').replace('&', '')
    clauses = re.findall(r'\((.*?)\)', raw_output)

    parsed_clauses = []
    for clause in clauses:
        clause = clause.replace('(', '').replace(')', '')
        # Split each clause into literals separated by 'or' (\\lor)
        literals = clause.split('\\lor')
        parsed_clause = []
        for literal in literals:
            literal = literal.strip()
            # Check if the literal is negated
            if literal.startswith('\\neg'):
                item = literal[4:].strip()  # Remove '\\neg ' prefix
                # item = find_best_match(item, sample_dict['menu_items'])
                parsed_clause.append(-item_to_number[item.lower()])
            else:
                item = literal.lower()
                # item = find_best_match(literal.lower(), sample_dict['menu_items'])
                parsed_clause.append(item_to_number[item])
        parsed_clauses.append(parsed_clause)

    return parsed_clauses


if __name__ == "__main__":
    model_name = 'gpt-4'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro
    ablation = ''  # '', 'sat', 'translate'
    append = '_2sat'  # '', '_2sat'

    file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()

    map2title = {'gpt-4': 'GPT-4', 'gpt-3.5': 'GPT-3.5', 'llama-2-70b': 'Llama-2-70B',
                 'text-bison@002': 'PaLM 2 (text-bison)', 'gemini-pro': 'Gemini Pro',
                 'llama-2-13b': 'Llama-2-13B'  }
    model_name = map2title[model_name]
    ablation = 'menu' if ablation == '' else ablation
    plot_path = f'plots/{model_name}/{ablation}/'
    os.makedirs(plot_path, exist_ok=True)
    correct = 0
    dataframe_dict = {'num_variables':[], 'num_clauses': [], 'alpha':[], 'is_sat':[],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                      'pred_is_sat': []}
    if append != '_2sat':
        dataframe_dict['model_count'] = []

    for id, line in enumerate(file_lines):
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

        # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
        #     continue

        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
        if append != '_2sat':
            dataframe_dict['model_count'].append(sample_dict['model_count'])

        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)

        if ablation == 'menu':
            try:
                orderable, not_orderable = \
                    parse_generated_output(raw_output=sample_dict['gpt_out'])
            except AttributeError:
                orderable, not_orderable = [], []

            if not sample_dict['is_sat']:  # if unsat
                if len(orderable) == 0 and len(not_orderable) == 0\
                        :  # gpt predicts unsat
                    dataframe_dict['correct'].append(True)
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                else:
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(True)  # pred sat
            else:  # if sat, verify solution
                if len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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
                                        assignment=assignment) and len(assignment) > 0
                    if verified:
                        dataframe_dict['correct'].append(True)
                    else:
                        dataframe_dict['correct'].append(False)
                # check if sat is predicted as sat or unsat
                if len(orderable) == 0 and len(not_orderable) == 0:
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                else:
                    dataframe_dict['pred_is_sat'].append(True)  # pred unsat

        elif ablation == 'sat':
            assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
            # only use those variables in the assignment which are part of the formula
            # sometimes the LLM creates and assigns values to new variables
            assignment = list(set(assignment).intersection(np.arange(1,sample_dict['num_vars']+1,1)))
            if not sample_dict['is_sat']:   # if unsat
                if len(assignment) == 0:
                    dataframe_dict['correct'].append(True)
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                else:
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(True)  # pred sat
            else:  # if sat, verify solution
                verified = \
                    verify_solution(num_vars=sample_dict['num_vars'],
                                    formula=sample_dict['formula'],
                                    assignment=assignment) and len(assignment) > 0
                if verified:
                    dataframe_dict['correct'].append(True)
                else:
                    dataframe_dict['correct'].append(False)
                # check if sat is predicted as sat or unsat
                if len(assignment) > 0:
                    dataframe_dict['pred_is_sat'].append(True)  # pred sat
                else:
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat

        elif ablation == 'translate':
            menu_items = sample_dict['menu_items']
            item_nums = np.arange(1, len(menu_items)+1)
            item2num = dict(zip(menu_items, item_nums))
            try:
                gen_formula = parse_generated_output_translate(raw_output=sample_dict['gpt_out'],
                                                           item_to_number=item2num)
                # verify isomorphism of generated formula
                pred_true = list(zip(gen_formula, sample_dict['formula']))
                formula_comparison = \
                    [True if set(pred_clause) == set(true_clause) else False for pred_clause, true_clause in pred_true]
                verified = True if sum(formula_comparison) == len(sample_dict['formula']) else False
                if verified:
                    dataframe_dict['correct'].append(True)
                else:
                    dataframe_dict['correct'].append(True)
            except KeyError:
                dataframe_dict['correct'].append(False)
            # bogus code
            if sample_dict['is_sat']:
                dataframe_dict['pred_is_sat'].append(True)
            else:
                dataframe_dict['pred_is_sat'].append(False)

        else:
            raise NotImplemented

    # Creating the DataFrame
    df = pd.DataFrame(dataframe_dict)
    # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
    accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
    lambda x: np.mean(x['correct'])).reset_index(name='accuracy')

    # Plotting
    # plt.interactive(False)
    plt.figure(figsize=(10, 6))
    for num_vars in accuracy_df['num_variables'].unique():
        # if num_vars in [4,6,10]:
            subset = accuracy_df[accuracy_df['num_variables'] == num_vars]
            plt.plot(subset['alpha'], subset['accuracy'], marker='o', label=f'Variables: {num_vars}')

    plt.xlabel('alpha', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=13)
    plt.xticks(fontsize=13)
    plt.legend()
    plt.title(f'{model_name}', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/all_alpha{append}.png')

    accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_df_simple['alpha'], accuracy_df_simple['accuracy'], marker='o')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlabel('alpha', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f'{model_name}', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/mean_alpha{append}.png')

    if append != '_2sat':
        model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 20)
        model_count_df = model_count_df.groupby('model_count')['correct'].mean().reset_index(name='accuracy')
        plt.figure(figsize=(10, 6))
        plt.plot(model_count_df['model_count'], model_count_df['accuracy'], marker='o')
        # plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('model count', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(f'{model_name}', fontsize=18)
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{plot_path}/model_count{append}.png')

    plt.figure(figsize=(10, 6))
    high_alpha_df = df[df['alpha'] >= 0][["pred_is_sat", "is_sat"]]
    labels = ['unSAT', 'SAT']
    # Generating the confusion matrix
    conf_matrix = confusion_matrix(high_alpha_df['pred_is_sat'], high_alpha_df['is_sat'])
    # Plotting the confusion matrix using seaborn
    heatmap = sns.heatmap(conf_matrix, annot=False, cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    # Get the color bar
    cbar = heatmap.collections[0].colorbar

    # Remove the labels from the color bar
    cbar.set_ticks([])
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    # Increase font size of the tick labels
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f'{model_name}', fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/cf{append}.png')

    # _3d_df_simple = df.groupby('num_variables')['correct'].mean().reset_index(name='accuracy')
    # plt.figure(figsize=(10, 6))
    # plt.plot(_3d_df_simple['num_variables'], _3d_df_simple['accuracy'], marker='o')
    # plt.xlabel('# variables')
    # plt.ylabel('accuracy')
    # plt.title(f'{model_name} accuracy vs # variables')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    _3d_df_simple = df.groupby('num_clauses')['correct'].mean().reset_index(name='accuracy')
    plt.figure(figsize=(10, 6))
    plt.plot(_3d_df_simple['num_clauses'], _3d_df_simple['accuracy'], marker='o')
    plt.xlabel('# clauses', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(f'{model_name}', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/clauses{append}.png')

    # correlation between prompt_tokens and completion_tokens
    corr_data = df[['num_prompt_tokens', 'num_completion_tokens']]
    correlation = corr_data.corr()
    # Performing linear regression
    slope, intercept = np.polyfit(corr_data['num_prompt_tokens'], corr_data['num_completion_tokens'], 1)
    line = slope * np.array(corr_data['num_prompt_tokens']) + intercept
    # Plotting the correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(corr_data['num_prompt_tokens'], corr_data['num_completion_tokens'])
    plt.plot(corr_data['num_prompt_tokens'], line, color='red', label='Fit Line: y={:.2f}x+{:.2f}'.format(slope, intercept))
    plt.title(f'{model_name}', fontsize=18)
    plt.xlabel('# prompt tokens', fontsize=16)
    plt.ylabel('# completion tokens', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.yticks(np.arange(0, 4100, 500))
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/corr_tokens{append}.png')

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


