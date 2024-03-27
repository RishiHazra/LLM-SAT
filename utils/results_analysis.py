import ast
import os.path
import re
from typing import List, Tuple, Dict, Optional

import Levenshtein as lev
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from verify_solution import verify_solution
from data_analysis import plot_dataset_analysis


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


def parse_generated_output_sat(raw_output: str) -> Optional[List]:
    # Regex pattern to match the specific format with numbers and boolean values

    pattern = r"<output:\s*\{\s*(?:'?\-?\d+'?:\s*(True|False),?\s*)*\}>"

    # pattern1 = r"<output:\s*{\s*(?:(-?\d+): (True|False),?\s*)*}>"
    # pattern2 = r"<output:\s*\{\s*(?:(?:'?\d+'?|\"?\d+\"?):\s*(True|False),?\s*)+\}>"

    # Search for the pattern in the string
    match = re.search(pattern, raw_output)
    # if match is None:
    #     match = re.search(pattern2, raw_output)

    # If a match is found, process it
    if match:
        # Find all key-value pairs in the matched string
        kv_pairs = re.findall(r"(-?\d+): (True|False)", match.group(0))
        if len(kv_pairs) == 0:
            kv_pairs = [(k.strip("''"), v) for k,v in re.findall(r"('?\d+'?): (True|False)", match.group(0))]

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
    return None


def parse_generated_output(raw_output: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    # Regular expression to find the lists
    pattern1 = r"```python\s+orderable\s*=\s*(\[[^\]]*\])\s*(,|\n)\s*not_orderable\s*=\s*(\[[^\]]*\])\s*```"
    match1 = re.search(pattern1, raw_output)

    pattern2 = r"orderable\s*=\s*(\[[^\]]*\])\s*(,|\n)\s*not_orderable\s*=\s*(\[[^\]]*\])\s*"
    match2 = re.search(pattern2, raw_output)

    pattern3 = r"(?i)(orderable\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*))\s*(?:,|\n)?\s*(not[_ ]orderable)\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*)"
    match3 = [match for match in re.finditer(pattern3, raw_output, re.DOTALL)]

    pattern4 = r"(?i)(orderable\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*))|((not[_ ]orderable)\s*[:=]\s*(\[[^\]]*\]|\S+(?:,\s*\S+)*))"
    match4 = [match for match in re.finditer(pattern4, raw_output, re.DOTALL)]

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
    elif match3:
        match = match3[-1]
        orderable_str = match.group(2).strip('[]')
        not_orderable_str = match.group(4).strip('[]')
        orderable_list = [item.strip(" '\"") for item in orderable_str.split(',')]
        not_orderable_list = [item.strip(" '\"") for item in not_orderable_str.split(',')]
    elif match4:
        match = match4[-1]
        orderable_str = match.group(2).strip('[]')
        not_orderable_str = match.group(4).strip('[]')
        orderable_list = [item.strip(" '\"") for item in orderable_str.split(',')]
        not_orderable_list = [item.strip(" '\"") for item in not_orderable_str.split(',')]
    else:
        orderable_list = not_orderable_list = None
        return orderable_list, not_orderable_list

    # postprocess
    orderable_list = [item.strip().strip("''").strip('""').lower() for item in orderable_list]
    not_orderable_list = [item.strip().strip("''").strip('""').lower() for item in not_orderable_list]
    return orderable_list, not_orderable_list


# Further refined function to correctly parse the CNF formula from LaTeX format
def parse_generated_output_translate(raw_output: str, item_to_number: Dict[str, int],
                                     menu_items: List) -> List[List]:
    # Extract the clauses from the LaTeX formatted string
    # Remove all LaTeX formatting and split by 'and' (\\land)
    raw_output = raw_output.replace('lnot', 'neg').replace('wedge', 'land').replace('vee', 'lor').replace('\\text{',
                                                                                                          '').replace(
        '}', '').replace(r'\\', '')
    raw_output = raw_output.replace('\\\\\n&', '').replace('&', '')
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
                item = find_best_match(literal.lower(), menu_items)
                parsed_clause.append(item_to_number[item])
        parsed_clauses.append(parsed_clause)

    return parsed_clauses


if __name__ == "__main__":
    # plot_dataset_analysis()
    model_name = 'gpt-4'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
    ablation = ''  # '', 'sat', 'translate'
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'

    file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}_mc.log', 'r').readlines()

    map2title = {'gpt-4': 'GPT-4', 'gpt-3.5': 'GPT-3.5', 'llama-2-70b': 'Llama-2-70B',
                 'text-bison@002': 'PaLM 2 (text-bison)', 'gemini-pro': 'Gemini Pro',
                 'llama-2-13b': 'Llama-2-13B', 'mixtral': 'Mixtral'}
    model_name = map2title[model_name]
    ablation = 'menu' if ablation == '' else ablation
    plot_path = f'plots/{model_name}/{ablation}/'
    os.makedirs(plot_path, exist_ok=True)
    correct = 0
    dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                      'pred_is_sat': []}
    if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name not in ['Llama-2-13B']:
        dataframe_dict['model_count'] = []
        dataframe_dict['satisfiability_ratio'] = []

    for id, line in enumerate(file_lines):
        print(id)
        #llama2-70b-menu-3shot: [1153, 1336, 2882, 3031, 3301]
        #mixtral-menu: [3024]
        # if id in [3024]:
        #     continue
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

        # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
        #     continue

        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
        if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name not in ['Llama-2-13B']:
            dataframe_dict['model_count'].append(sample_dict['model_count'])
            dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)

        if ablation == 'menu':
            try:
                orderable, not_orderable = \
                    parse_generated_output(raw_output=sample_dict['gpt_out'])
            except AttributeError:
                orderable, not_orderable = None, None

            if not sample_dict['is_sat']:  # if unsat
                if orderable is None and not_orderable is None:  # inconclusive output/solution
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(np.nan)
                elif len(orderable) == 0 and len(not_orderable) == 0 :  # gpt predicts unsat
                    dataframe_dict['correct'].append(True)
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                else:
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(True)  # pred sat
            else:  # if sat, verify solution
                if orderable is None and not_orderable is None:  # inconclusive output/solution
                    dataframe_dict['correct'].append(False)
                elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
                    dataframe_dict['correct'].append(False)
                else:
                    assignment = []
                    for item in orderable:
                        item = find_best_match(item, sample_dict['menu_items'])
                        assignment.append(sample_dict['menu_items'].index(item) + 1)
                    for item in not_orderable:
                        item = find_best_match(item, sample_dict['menu_items'])
                        assignment.append(-sample_dict['menu_items'].index(item) - 1)
                    verified = \
                        verify_solution(num_vars=sample_dict['num_vars'],
                                        formula=sample_dict['formula'],
                                        assignment=assignment) and len(assignment) > 0
                    if verified:
                        dataframe_dict['correct'].append(True)
                    else:
                        dataframe_dict['correct'].append(False)
                # check if sat is predicted as sat or unsat
                if orderable is None and not_orderable is None:  # inconclusive output/solution
                    dataframe_dict['pred_is_sat'].append(np.nan)
                elif len(orderable) == 0 and len(not_orderable) == 0:
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                else:
                    dataframe_dict['pred_is_sat'].append(True)  # pred unsat

        elif ablation == 'sat':
            assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
            # only use those variables in the assignment which are part of the formula
            # sometimes the LLM creates and assigns values to new variables
            if assignment:
                assignment = list(set(assignment).intersection(np.arange(1, sample_dict['num_vars'] + 1, 1)))
            if not sample_dict['is_sat']:  # if unsat
                if assignment is None:
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(np.nan)
                elif len(assignment) == 0:
                    dataframe_dict['correct'].append(True)
                    dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                else:
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(True)  # pred sat
            else:  # if sat, verify solution
                if assignment is None:
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(np.nan)
                else:
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
            item_nums = np.arange(1, len(menu_items) + 1)
            item2num = dict(zip(menu_items, item_nums))
            try:
                gen_formula = parse_generated_output_translate(raw_output=sample_dict['gpt_out'],
                                                               item_to_number=item2num,
                                                               menu_items=menu_items)
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
    # analyze the dataset
    # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
    accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
        lambda x: np.mean(x['correct'])).reset_index(name='accuracy')

    # Plotting
    # plt.interactive(False)
    few_shot_title = {'': '0-shot', '_3shot': '3-shot'}
    plt.figure(figsize=(10, 6))
    window_size = 3  # This is the number of points to include in the moving average
    for num_vars in accuracy_df['num_variables'].unique():
        # if num_vars in [4,6,10]:
        subset = accuracy_df[accuracy_df['num_variables'] == num_vars]
        plt.plot(subset['alpha'],
                 subset['accuracy'],
                 marker='o', label=f'Variables: {num_vars}')

    plt.xlabel('alpha', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=13)
    plt.xticks(fontsize=13)
    plt.legend()
    plt.title(f'{model_name} ({few_shot_title[few_shot]})', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/all_alpha{append}{few_shot}.png')

    accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
    plt.figure(figsize=(10, 6))
    window_size = 4  # This is the number of points to include in the moving average
    plt.plot(accuracy_df_simple['alpha'],
             accuracy_df_simple['accuracy'].rolling(window=window_size).mean(),
             marker='o')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('alpha', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f'{model_name} ({few_shot_title[few_shot]})', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/mean_alpha{append}{few_shot}.png')

    if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name not in ['Llama-2-13B']:
        model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 20)
        model_count_df = model_count_df.groupby('satisfiability_ratio')['correct'].mean().reset_index(name='accuracy')
        plt.figure(figsize=(10, 6))
        window_size = 3
        plt.plot(model_count_df['satisfiability_ratio'],
                 model_count_df['accuracy'].rolling(window=window_size).median(), marker='o')
        # plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('satisfiability ratio (log-scale)', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xscale("log")
        plt.title(f'{model_name}', fontsize=18)
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{plot_path}/model_count{append}{few_shot}.png')

    # 3D plot
    # model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 20)
    model_count_df = df.groupby(['alpha', 'num_variables'])['correct'].mean().reset_index(name='accuracy')
    azimuth = 80  # The angle to rotate around the z-axis
    elevation = 30
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    model_count_df['rolling_avg_accuracy'] = model_count_df['accuracy'].rolling(window=3).mean().fillna(method='bfill')
    # We need to create a regular grid where each model_count and num_vars are represented
    # Let's create an interpolation grid for the model_count and num_vars values
    alpha_i = np.linspace(model_count_df['alpha'].min(),
                                model_count_df['alpha'].max(),
                                len(model_count_df['alpha'].unique()))
    num_vars_i = np.linspace(model_count_df['num_variables'].min(),
                             model_count_df['num_variables'].max(),
                             len(model_count_df['num_variables'].unique()))
    alpha_ii, num_vars_ii = np.meshgrid(alpha_i, num_vars_i)

    # Interpolating; this will fill in the gaps in the rolling_avg_accuracy on the new grid
    accuracy_i = griddata((model_count_df['alpha'],
                           model_count_df['num_variables']),
                          model_count_df['rolling_avg_accuracy'],
                          (alpha_ii, num_vars_ii), method='cubic')

    surf = ax.plot_surface(alpha_ii, num_vars_ii, accuracy_i, cmap='viridis', edgecolor='none')
    ax.view_init(elev=elevation, azim=azimuth)

    ax.zaxis.set_tick_params(length=0)

    # Increase the space between the z-axis title and the ticks
    ax.zaxis.labelpad = 30
    # Remove the black tick lines for all axes
    ax.xaxis.line.set_lw(0.)
    ax.yaxis.line.set_lw(0.)
    ax.zaxis.line.set_lw(0.)

    # Move the ticks to the left of the z-axis
    ax.zaxis.set_tick_params(pad=15)

    # # Setting the new limits
    ax.set_xlim(model_count_df['alpha'].max(), model_count_df['alpha'].min())
    ax.set_ylim(model_count_df['num_variables'].max(), model_count_df['num_variables'].min())

    # Customize the z axis.
    ax.set_zlim(0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customizing the axes tick labels
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.grid(True)
    ax.set_ylabel('# variables', fontsize=15, labelpad=20)
    ax.set_xlabel('alpha', fontsize=15, labelpad=20)
    ax.set_zlabel('accuracy', fontsize=15, labelpad=30)
    # Rotate the z-axis label
    ax.zaxis.set_rotate_label(False)  # This disables automatic rotation
    ax.zaxis.label.set_rotation(90)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/accuracy_3d_{append}{few_shot}.png')
    # plt.show()

    # 3D plot another
    model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 0)
    # model_count_df['satisfiability_ratio'] = np.log(model_count_df['satisfiability_ratio'])
    model_count_df = model_count_df.groupby(['alpha', 'satisfiability_ratio'])['correct'].mean().reset_index(name='accuracy')
    azimuth = -30  # The angle to rotate around the z-axis
    # elevation = 0
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    model_count_df['rolling_avg_accuracy'] = model_count_df['accuracy'].rolling(window=6).mean().fillna(method='bfill')
    # We need to create a regular grid where each model_count and num_vars are represented
    # Let's create an interpolation grid for the model_count and num_vars values
    alpha_i = np.linspace(model_count_df['alpha'].min(),
                          model_count_df['alpha'].max(),
                          len(model_count_df['alpha'].unique()))
    sat_i = np.linspace(model_count_df['satisfiability_ratio'].min(),
                             model_count_df['satisfiability_ratio'].max(),
                             len(model_count_df['satisfiability_ratio'].unique()))
    alpha_ii, sat_ii = np.meshgrid(alpha_i, sat_i)

    # Interpolating; this will fill in the gaps in the rolling_avg_accuracy on the new grid
    accuracy_i = griddata((model_count_df['satisfiability_ratio'],
                           model_count_df['alpha']),
                          model_count_df['rolling_avg_accuracy'],
                          (sat_ii, alpha_ii), method='linear')

    surf = ax.plot_surface(sat_ii, alpha_ii, accuracy_i, cmap='viridis', edgecolor='none')
    ax.view_init(azim=azimuth)

    # ax.zaxis.set_tick_params(length=0)

    # Increase the space between the z-axis title and the ticks
    # ax.zaxis.labelpad = 30
    # Remove the black tick lines for all axes
    # ax.xaxis.line.set_lw(0.)
    # ax.yaxis.line.set_lw(0.)
    # ax.zaxis.line.set_lw(0.)

    # Move the ticks to the left of the z-axis
    # ax.zaxis.set_tick_params(pad=15)

    # # Setting the new limits
    # ax.set_xlim(model_count_df['alpha'].max(), model_count_df['alpha'].min())
    # ax.set_ylim(model_count_df['satisfiability_ratio'].max(),
    #             model_count_df['satisfiability_ratio'].min())

    # Customize the z axis.
    # ax.set_xscale("log")
    ax.set_xlim(model_count_df['satisfiability_ratio'].max(), model_count_df['satisfiability_ratio'].min())
    ax.set_zlim(0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customizing the axes tick labels
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.grid(True)
    ax.set_ylabel('alpha', fontsize=15, labelpad=20)
    ax.set_xlabel('satisfiability ratio', fontsize=15, labelpad=20)
    ax.set_zlabel('accuracy', fontsize=15, labelpad=30)
    # Rotate the z-axis label
    # ax.zaxis.set_rotate_label(False)  # This disables automatic rotation
    # ax.zaxis.label.set_rotation(90)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/accuracy_3d_{append}{few_shot}_2.png')


    plt.figure(figsize=(10, 6))
    high_alpha_df = df[df['alpha'] >= 0][["pred_is_sat", "is_sat"]]
    high_alpha_df = high_alpha_df.dropna()
    labels = ['unSAT', 'SAT']
    # Generating the confusion matrix
    conf_matrix = confusion_matrix(high_alpha_df['pred_is_sat'].astype('bool'), high_alpha_df['is_sat'])
    column_sums = conf_matrix.sum(axis=0)
    normalized_conf_matrix = conf_matrix / column_sums[np.newaxis, :]
    # Plotting the confusion matrix using seaborn
    heatmap = sns.heatmap(normalized_conf_matrix, annot=True, cmap='Blues',
                          xticklabels=labels, yticklabels=labels,
                          fmt='.2f', annot_kws={"size": 18})
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
    plt.savefig(f'{plot_path}/cf{append}{few_shot}.png')

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
    window_size = 3
    plt.plot(_3d_df_simple['num_clauses'],
             _3d_df_simple['accuracy'].rolling(window=window_size).mean(), marker='o')
    plt.xlabel('# clauses', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(f'{model_name}', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/clauses{append}{few_shot}.png')

    # correlation between prompt_tokens and completion_tokens
    corr_data = df[['num_prompt_tokens', 'num_completion_tokens']]
    correlation = corr_data.corr()
    # Performing linear regression
    slope, intercept = np.polyfit(corr_data['num_prompt_tokens'], corr_data['num_completion_tokens'], 1)
    line = slope * np.array(corr_data['num_prompt_tokens']) + intercept
    # Plotting the correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(corr_data['num_prompt_tokens'], corr_data['num_completion_tokens'])
    plt.plot(corr_data['num_prompt_tokens'], line, color='red',
             label='Fit Line: y={:.2f}x+{:.2f}'.format(slope, intercept))
    plt.title(f'{model_name}', fontsize=18)
    plt.xlabel('# prompt tokens', fontsize=16)
    plt.ylabel('# completion tokens', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.yticks(np.arange(0, 4100, 500))
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/corr_tokens{append}{few_shot}.png')

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
