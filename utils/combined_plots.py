import ast
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sklearn.metrics import r2_score, confusion_matrix

from results_analysis import parse_generated_output, find_best_match, parse_generated_output_sat, \
    parse_generated_output_translate
from utils import map2title
from verify_solution import verify_solution


def read_file_lines(model_name, ablation, append):
    try:
        file_path = f'../out_data/{model_name}{ablation}/data_log{append}_mc.log'
        with open(file_path, 'r') as file:
            return file.readlines()
    except FileNotFoundError:
        file_path = f'../out_data/{model_name}{ablation}/data_log{append}.log'
        with open(file_path, 'r') as file:
            return file.readlines()


def process_file_lines(file_lines, model_name, ablation):
    dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                      'pred_is_sat': []}
    for id, line in enumerate(file_lines):
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])
        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)
        # Additional processing based on ablation
        if ablation == 'menu':
            try:
                orderable, not_orderable = parse_generated_output(raw_output=sample_dict['gpt_out'])
            except AttributeError:
                orderable, not_orderable = None, None

            if orderable is None and not_orderable is None:
                for key in ['num_variables', 'num_clauses', 'is_sat', 'num_prompt_tokens', 'num_completion_tokens',
                            'alpha']:
                    dataframe_dict[key].pop()
                print(model_name, "Discarding ...")
                continue
                # orderable, not_orderable = None, None

            if not sample_dict['is_sat']:  # if unsat
                if orderable is None and not_orderable is None:  # inconclusive output/solution
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(np.nan)
                elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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
            try:
                assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
            except TypeError:
                assignment = None
            if assignment is None:
                dataframe_dict['num_variables'].pop()
                dataframe_dict['num_clauses'].pop()
                dataframe_dict['is_sat'].pop()
                dataframe_dict['num_prompt_tokens'].pop()
                dataframe_dict['num_completion_tokens'].pop()
                dataframe_dict['alpha'].pop()
                print(model_name, "Discarding ...")
                continue
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
    return dataframe_dict


def compute_confidence_intervals(grouped):
    means = grouped.mean()
    counts = grouped.count()
    # Binomial standard error
    stderr = np.sqrt(means * (1 - means) / counts)
    # 90% confidence intervals
    z = 1.440  # 1.645 # 1.96
    lower = means - z * stderr
    upper = means + z * stderr
    return lower, upper


def acc_combined():
    sat_class = '_2sat'  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.axvspan(3.2, 7.9, color='gray', alpha=0.5)
    # all_model_list = ['gpt-3.5', 'gpt-4', 'llama-2-70b', 'text-bison@002', 'gemini-pro', 'mixtral']
    all_model_list = ['claude_37_sonnet', 'gpt-4o', 'gemini_20', 'deepseek_v3',
                      'deepseek']  # 'gemini_25_pro', 'deepseek_v3'
    # all_model_list = ['distilled_qwen_15', 'distilled_qwen_7', 'distilled_qwen_14', 'distilled_qwen_32']
    few_shot_title = {'': '0-shot', '_3shot': '3-shot'}

    colors = {
        'Claude 3.7 Sonnet': '#FF5733',  # Coral orange
        'GPT-4o': '#4A90E2',  # Bright blue
        'Gemini 2.0 Flash': '#50C878',  # Emerald green
        'DeepSeek V3': '#9B59B6',  # Royal purple
        'DeepSeek R1': '#2C3E50',  # Dark slate
        'Gemini 2.5 Pro': '#F1C40F',  # Golden yellow
    }
    # colors = {
    #     'R1 Distilled Qwen 1.5B': '#4A90E2',  # Sky Blue
    #     'R1 Distilled Qwen 7B': '#F45B69',  # Coral Red
    #     'R1 Distilled Qwen 14B': '#50C878',  # Emerald Green
    #     'R1 Distilled Qwen 32B': '#F4C542',  # Goldenrod Yellow
    #     'DeepSeek V3': '#9B59B6',  # Royal Purple
    # }

    for model_name in all_model_list:
        ablation = ''  # '', 'sat', 'translate'
        file_lines = read_file_lines(model_name, ablation, sat_class)
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)

        if model_name == 'GPT-4o' and ablation == 'menu' and sat_class != '_2sat':
            file_lines = file_lines[1000:]

        dataframe_dict = process_file_lines(file_lines, model_name, ablation)

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
        # noise
        # if model_name == 'R1 Distilled Qwen 32B':
        #     accuracy_df_simple.loc[accuracy_df_simple['alpha'] >= 1, 'accuracy'] += 0.11
        window_size = 6  # This is the number of points to include in the moving average

        accuracy_df_simple['rolling_mean'] = accuracy_df_simple['accuracy'].rolling(window=window_size).mean().tolist()
        # accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]

        # confidence intervals
        lower, upper = compute_confidence_intervals(grouped=df.groupby('alpha')['correct'])
        accuracy_df_simple['rolling_upper'] = upper.rolling(window=window_size).mean().clip(upper=1).tolist()
        accuracy_df_simple['rolling_lower'] = lower.rolling(window=window_size).mean().clip(lower=0).tolist()

        # accuracy_df_simple['std'] = df.groupby('alpha')['correct'].std().tolist()
        # accuracy_df_simple['rolling_std'] = accuracy_df_simple['std'].rolling(window=window_size).mean().tolist()
        accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]

        ax = accuracy_df_sliced.plot(ax=ax, x='alpha',
                                     y='rolling_mean',
                                     kind='line', label=model_name, linewidth=3,
                                     color=colors[model_name])
        ax.fill_between(accuracy_df_sliced['alpha'],
                        accuracy_df_sliced['rolling_lower'],
                        accuracy_df_sliced['rolling_upper'], alpha=0.2,
                        color=colors[model_name])

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axvline(x=4.24, color='lime', linestyle='--', linewidth=4)
    plt.title(f'3-SAT (SAT-CNF)', fontsize=20)
    # plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.show()
    # plt.savefig(f'{plot_path}/combined_mean_alpha{append}{few_shot}.png')


def model_count_alpha_combined():
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    # append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
    fig, ax = plt.subplots(figsize=(10, 8))

    all_model_list = ['deepseek_v3', 'deepseek']  # , 'gpt-4o', 'gemini_20', 'deepseek']
    colors = {
        'Claude 3.7 Sonnet': '#FF5733',  # Coral orange
        'GPT-4o': '#4A90E2',  # Bright blue
        'Gemini 2.0 Flash': '#50C878',  # Emerald green
        'DeepSeek V3': '#9B59B6',  # Royal purple
        'DeepSeek R1': '#2C3E50'  # Dark slate
    }

    for model_name in all_model_list:
        ablation = ''  # '', 'sat', 'translate'
        file_lines = open(f'../output_mc/{model_name}{ablation}/data_log{append}{few_shot}_mc.log', 'r').readlines()
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [], 'correct': [],
                          'num_prompt_tokens': [], 'num_completion_tokens': [], 'pred_is_sat': [], 'model_count': [],
                          'satisfiability_ratio': []}
        # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':

        if model_name == 'GPT-4o' and ablation == 'menu' and few_shot != '_3shot':
            file_lines = file_lines[1000:]

        for id, line in enumerate(file_lines):
            print(id)
            # if few_shot == '' and id in [3024]:
            #     continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
            dataframe_dict['model_count'].append(sample_dict['model_count'])
            dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

            alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
            dataframe_dict['alpha'].append(alpha)

            if ablation == 'menu':
                try:
                    orderable, not_orderable = \
                        parse_generated_output(raw_output=sample_dict['gpt_out'])
                except AttributeError:
                    dataframe_dict['num_variables'].pop()
                    dataframe_dict['num_clauses'].pop()
                    dataframe_dict['is_sat'].pop()
                    dataframe_dict['num_prompt_tokens'].pop()
                    dataframe_dict['num_completion_tokens'].pop()
                    dataframe_dict['alpha'].pop()
                    dataframe_dict['model_count'].pop()
                    dataframe_dict['satisfiability_ratio'].pop()
                    print("Discarding ...")
                    continue
                    # orderable, not_orderable = None, None

                if not sample_dict['is_sat']:  # if unsat
                    if orderable is None and not_orderable is None:  # inconclusive output/solution
                        dataframe_dict['correct'].append(False)
                        dataframe_dict['pred_is_sat'].append(np.nan)
                    elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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
                try:
                    assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
                except TypeError:
                    assignment = None
                if assignment is None:
                    dataframe_dict['num_variables'].pop()
                    dataframe_dict['num_clauses'].pop()
                    dataframe_dict['is_sat'].pop()
                    dataframe_dict['num_prompt_tokens'].pop()
                    dataframe_dict['num_completion_tokens'].pop()
                    dataframe_dict['alpha'].pop()
                    dataframe_dict['model_count'].pop()
                    dataframe_dict['satisfiability_ratio'].pop()
                    print("Discarding ...")
                    continue
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
        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        df = df[df['is_sat'] == True]
        model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 20)

        easy_df = model_count_df[(model_count_df['alpha'] <= 3.2) | (model_count_df['alpha'] >= 7.9)].groupby(
            ['satisfiability_ratio'])['correct'].mean().reset_index(name='accuracy')
        easy_df['rolling_avg_accuracy'] = easy_df['accuracy'].rolling(window=4).mean().fillna(method='bfill')
        hard_df = model_count_df[(model_count_df['alpha'] < 7.9) & (model_count_df['alpha'] > 3.2)].groupby(
            ['satisfiability_ratio'])['correct'].mean().reset_index(name='accuracy')
        hard_df['rolling_avg_accuracy'] = hard_df['accuracy'].rolling(window=5).mean().fillna(method='bfill')

        # confidence intervals
        lower, upper = compute_confidence_intervals(grouped=model_count_df[(model_count_df['alpha'] <= 3.2) | (model_count_df['alpha'] >= 7.9)].groupby(
            ['satisfiability_ratio'])['correct'])
        easy_df['rolling_upper'] = upper.rolling(window=4).mean().clip(upper=1).tolist()
        easy_df['rolling_lower'] = lower.rolling(window=4).mean().clip(lower=0).tolist()

        lower, upper = compute_confidence_intervals(
            grouped=model_count_df[(model_count_df['alpha'] < 7.9) & (model_count_df['alpha'] > 3.2)].groupby(
            ['satisfiability_ratio'])['correct'])
        hard_df['rolling_upper'] = upper.rolling(window=5).mean().clip(upper=1).tolist()
        hard_df['rolling_lower'] = lower.rolling(window=5).mean().clip(lower=0).tolist()

        # accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]
        # accuracy_df_simple['rolling_std'] = accuracy_df_simple['std'].rolling(window=window_size).mean().tolist()
        # accuracy_df_simple['rolling_min'] = accuracy_df_simple['accuracy'].rolling(window=window_size).min().tolist()
        # accuracy_df_simple['rolling_max'] = accuracy_df_simple['accuracy'].rolling(window=window_size).max().tolist()

        ax = easy_df.plot(ax=ax, x='satisfiability_ratio',
                          y='rolling_avg_accuracy',
                          kind='line', label=f'{model_name}-easy',
                          linewidth=3, linestyle='--', color=colors[model_name])
        ax.fill_between(easy_df['satisfiability_ratio'],
                        easy_df['rolling_lower'],
                        easy_df['rolling_upper'], alpha=0.2,
                        color=colors[model_name])
        ax = hard_df.plot(ax=ax, x='satisfiability_ratio',
                          y='rolling_avg_accuracy',
                          kind='line', label=f'{model_name}-hard',
                          linewidth=3, color=colors[model_name])
        ax.fill_between(hard_df['satisfiability_ratio'],
                        hard_df['rolling_lower'],
                        hard_df['rolling_upper'], alpha=0.2,
                        color=colors[model_name])

    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('satisfiability ratio', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(f'{model_name}', fontsize=22)
    # plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()
    # print(f'{plot_path}/combined_model_count_alpha.png')
    # plt.savefig(f'{plot_path}/combined_model_count_alpha.png')


def plot_token_relationship(df, input_col, output_col, title=None, plot_path=''):
    x = df[input_col].values
    y = df[output_col].values

    # Linear Fit
    slope, intercept, _, _, _ = linregress(x, y)
    linear_pred = intercept + slope * x
    linear_r2 = r2_score(y, linear_pred)

    # Quadratic Fit
    coeffs_2 = np.polyfit(x, y, deg=2)
    quad_pred = np.polyval(coeffs_2, x)
    quad_r2 = r2_score(y, quad_pred)

    # Cubic Fit
    coeffs_3 = np.polyfit(x, y, deg=3)
    cubic_pred = np.polyval(coeffs_3, x)
    cubic_r2 = r2_score(y, cubic_pred)

    # Determine Best Fit
    r2_scores = {'Linear': linear_r2, 'Quadratic': quad_r2, 'Cubic': cubic_r2}
    best_fit = max(r2_scores, key=r2_scores.get)

    # Plotting
    plt.scatter(x, y)

    # X-axis range for smooth lines
    x_vals = np.linspace(x.min(), x.max(), 200)

    # Plot fits
    plt.plot(x_vals, intercept + slope * x_vals, color='red', linestyle='--', label=f'Linear R²={linear_r2:.3f}')
    plt.plot(x_vals, np.polyval(coeffs_2, x_vals), color='green', linestyle='-.', label=f'Quadratic R²={quad_r2:.3f}')
    plt.plot(x_vals, np.polyval(coeffs_3, x_vals), color='yellow', linestyle=':', linewidth=5,
             label=f'Cubic R²={cubic_r2:.3f}')

    # Highlight best fit
    plt.title(title or f'Best Fit: {best_fit}', fontsize=20)
    plt.xlabel('# input tokens', fontsize=18)
    plt.ylabel('# output tokens', fontsize=18)
    plt.legend()
    # plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{plot_path}/out_vs_input_tokens.png')

    print(f"Best fit for {title or 'current dataset'}: {best_fit}")
    print(f"R² Scores -> Linear: {linear_r2:.4f}, Quadratic: {quad_r2:.4f}, Cubic: {cubic_r2:.4f}")


def output_input_token_corr():
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    # fig, ax = plt.subplots(figsize=(10, 8))

    all_model_list = ['deepseek', 'gpt-4o', 'claude_37_sonnet', 'gemini_20', 'deepseek_v3']
    few_shot_title = {'': '0-shot', '_3shot': '3-shot'}

    for model_name in all_model_list:
        fig, ax = plt.subplots(figsize=(10, 8))
        ablation = 'sat'  # '', 'sat', 'translate'
        try:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}_mc.log', 'r').readlines()
        except FileNotFoundError:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'num_prompt_tokens': [], 'num_completion_tokens': []}

        if model_name == 'GPT-4o' and ablation == 'menu':
            file_lines = file_lines[1000:]

        for id, line in enumerate(file_lines):
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        df = df[df['num_prompt_tokens'] <= 700]
        df = df[df['num_completion_tokens'] <= 20000]
        plot_token_relationship(df, 'num_prompt_tokens', 'num_completion_tokens',
                                title=model_name, plot_path=plot_path)

    # # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel(r'$ \alpha $', fontsize=20)
    # plt.ylabel('generated tokens', fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.title('Token count vs.' + r'$ \alpha $', fontsize=20)
    # # plt.grid(True)
    # plt.tight_layout()
    # plt.legend(fontsize=18)
    # # plt.show()
    # plt.savefig(f'{plot_path}/completion_tokens_accuracy.png')


def output_tokens_acc_corr():
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    # fig, ax = plt.subplots(figsize=(10, 8))

    all_model_list = ['claude_37_sonnet', 'gpt-4o', 'gemini_20', 'deepseek', 'deepseek_v3']
    few_shot_title = {'': '0-shot', '_3shot': '3-shot'}
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        'Claude 3.7 Sonnet': '#FF5733',  # Coral orange
        'GPT-4o': '#4A90E2',  # Bright blue
        'Gemini 2.0 Flash': '#50C878',  # Emerald green
        'DeepSeek V3': '#9B59B6',  # Royal purple
        'DeepSeek R1': '#2C3E50',  # Dark slate
        'Gemini 2.5 Pro': '#F1C40F',  # Golden yellow
    }

    for model_name in all_model_list:
        ablation = ''  # '', 'sat', 'translate'
        try:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}_mc.log', 'r').readlines()
        except FileNotFoundError:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}

        if model_name == 'GPT-4o' and ablation == 'menu':
            file_lines = file_lines[1000:]

        for id, line in enumerate(file_lines):
            if model_name == 'Mixtral' and id == 3024:
                continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
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
                    elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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
                try:
                    assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
                except TypeError:
                    assignment = None
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

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        accuracy_df_simple = df.groupby('alpha')['num_completion_tokens'].mean().reset_index(name='generated_tokens')
        window_size = 4
        accuracy_df_simple['rolling_mean'] = accuracy_df_simple['generated_tokens'].rolling(
            window=window_size).mean().tolist()
        accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]
        # fig, ax = plt.subplots(figsize=(10, 8))
        ax = accuracy_df_sliced.plot(ax=ax, x='alpha',
                                     y='rolling_mean',
                                     kind='line', linewidth=3, label=model_name,
                                     color=colors[model_name])

    # plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel('# output tokens', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Token count vs.' + r'$ \alpha $' + ' (SAT-Menu)', fontsize=20)
    # plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.show()
    # plt.savefig(f'{plot_path}/completion_tokens_accuracy.png')


def unsat_combined():
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    fig, ax = plt.subplots(figsize=(10, 6))

    all_model_list = ['claude_37_sonnet', 'gpt-4o', 'gemini_20', 'deepseek_v3', 'deepseek']

    colors = {
        'Claude 3.7 Sonnet': '#FF5733',  # Coral orange
        'GPT-4o': '#4A90E2',  # Bright blue
        'Gemini 2.0 Flash': '#50C878',  # Emerald green
        'DeepSeek V3': '#9B59B6',  # Royal purple
        'DeepSeek R1': '#2C3E50'  # Dark slate
    }

    # Calculate accuracy and error rates
    accuracy_rates = []
    error_rates = []

    for model_name in all_model_list:
        # model_name = 'llama-2-70b'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
        ablation = ''  # '', 'sat', 'translate'
        try:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}_mc.log', 'r').readlines()
        except FileNotFoundError:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}

        if model_name == 'GPT-4o' and ablation == 'menu' and append != '_2sat':
            file_lines = file_lines[1000:]

        for id, line in enumerate(file_lines):
            if model_name == 'Mixtral' and id == 3024:
                continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
            dataframe_dict['alpha'].append(alpha)

            if ablation == 'menu':
                try:
                    orderable, not_orderable = \
                        parse_generated_output(raw_output=sample_dict['gpt_out'])
                except AttributeError:
                    dataframe_dict['num_variables'].pop()
                    dataframe_dict['num_clauses'].pop()
                    dataframe_dict['is_sat'].pop()
                    dataframe_dict['num_prompt_tokens'].pop()
                    dataframe_dict['num_completion_tokens'].pop()
                    dataframe_dict['alpha'].pop()
                    print(model_name, "Discarding ...")
                    continue

                if not sample_dict['is_sat']:  # if unsat
                    if orderable is None and not_orderable is None:  # inconclusive output/solution
                        dataframe_dict['correct'].append(False)
                        dataframe_dict['pred_is_sat'].append(np.nan)
                    elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
        # Plotting
        # plt.interactive(False)
        few_shot_title = {'': '0-shot', '_3shot': '3-shot'}
        accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
        window_size = 4  # This is the number of points to include in the moving average
        # plt.plot(accuracy_df_simple['alpha'],
        #          accuracy_df_simple['accuracy'].rolling(window=window_size).mean(),
        #          marker='o', label=model_name)

        # plotting unSAT accuracy
        df = df[df['is_sat'] == False][["pred_is_sat", "is_sat"]]
        correct_predictions = df[df['pred_is_sat'] == df['is_sat']].shape[0]
        total_predictions = df.shape[0]
        accuracy = correct_predictions / total_predictions
        error = 1 - accuracy
        accuracy_rates.append(accuracy * 100)  # Convert to percentage
        error_rates.append(error * 100)  # Convert to percentage

    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel('alpha', fontsize=16)
    # plt.ylabel('accuracy', fontsize=16)
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.title(f'{few_shot_title[few_shot]} Performance', fontsize=18)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend(fontsize=16)
    # plt.savefig(f'{plot_path}/combined_mean_alpha{append}{few_shot}.png')

    # Create stacked bar plot
    # fig, ax = plt.subplots()
    mapped_model_list = [map2title[x] for x in all_model_list]
    index = np.arange(len(mapped_model_list))
    ax.bar(index, accuracy_rates, width=0.5, label='Correctly Predicted', color='skyblue')
    ax.bar(index, error_rates, width=0.5, bottom=accuracy_rates, label='Incorrectly Predicted', color='salmon')

    # Adding some aesthetics
    ax.set_ylabel('Percentage', fontsize=16)
    ax.set_title('unSAT Accuracy (SAT-Menu)', fontsize=18, pad=30)
    plt.yticks(fontsize=13)
    plt.xticks(index, mapped_model_list, rotation=45, fontsize=13)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=16)
    # plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plot_path}/combined_unsat_accuracy_{append}{few_shot}.png')
    # plt.show()


def few_shot_combined():
    fig, ax = plt.subplots(figsize=(10, 8))
    for few_shot in ['', '_3shot']:
        # if few_shot == '_3shot':
        #     few_shot = ''
        model_name = 'llama-2-70b'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
        ablation = ''  # ''
        append = ''  # '', '_2sat'
        # few_shot = ''  # '', '_3shot'
        # append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
        try:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}_mc.log', 'r').readlines()
        except FileNotFoundError:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}.log', 'r').readlines()

        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}
        # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
        #     dataframe_dict['model_count'] = []
        #     dataframe_dict['satisfiability_ratio'] = []

        for id, line in enumerate(file_lines):
            print(id)
            if few_shot == '' and id in [1153, 1154, 1336, 2882, 3031, 3301]:  # [1153, 1336, 2882, 3031, 3301] [3024]
                continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
            #     dataframe_dict['model_count'].append(sample_dict['model_count'])
            #     dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

            alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
            dataframe_dict['alpha'].append(alpha)

            try:
                orderable, not_orderable = \
                    parse_generated_output(raw_output=sample_dict['gpt_out'])
            except AttributeError:
                orderable, not_orderable = None, None

            if not sample_dict['is_sat']:  # if unsat
                if orderable is None and not_orderable is None:  # inconclusive output/solution
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(np.nan)
                elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
        accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
            lambda x: np.mean(x['correct'])).reset_index(name='accuracy')

        # Plotting
        # plt.interactive(False)
        few_shot_title = {'': '0-shot', '_3shot': '3-shot'}

        # window_size = 4
        accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
        window_size = 4  # This is the number of points to include in the moving average
        # acc_mean = accuracy_df_simple['correct'].mean()
        # accuracy_df_simple['acc_mean'] = acc_mean
        # accuracy_df_simple['std'] = df.groupby('alpha')['correct'].std().tolist()
        # accuracy_df_simple['min'] = df.groupby('alpha')['correct'].min().tolist()
        # accuracy_df_simple['max'] = df.groupby('alpha')['correct'].max().tolist()
        # accuracy_df_simple['rolling_std'] = accuracy_df_simple['std'].rolling(window=window_size).mean().tolist()
        # accuracy_df_simple['rolling_min'] = accuracy_df_simple['accuracy'].rolling(window=window_size).min().tolist()
        # accuracy_df_simple['rolling_max'] = accuracy_df_simple['accuracy'].rolling(window=window_size).max().tolist()
        accuracy_df_simple['rolling_mean'] = accuracy_df_simple['accuracy'].rolling(window=window_size).mean().tolist()

        accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]
        ax = accuracy_df_sliced.plot(ax=ax, x='alpha',
                                     y='rolling_mean',
                                     kind='line', label=few_shot_title[few_shot],
                                     linewidth=3)
        # ax.fill_between(accuracy_df_sliced['alpha'],
        #                 accuracy_df_sliced['rolling_mean'] - accuracy_df_sliced['rolling_std'],
        #                 accuracy_df_sliced['rolling_mean'] + accuracy_df_sliced['rolling_std'], alpha=0.2)
        # ax.fill_between(accuracy_df_sliced['alpha'],
        #                 accuracy_df_sliced['rolling_max'],
        #                 accuracy_df_sliced['rolling_min'], alpha=0.2)

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'{model_name}', fontsize=20)
    # plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/combined_mean_alpha{append}.png')


def decision_search_combined():
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.fill_between(, 3.2, 7.9, alpha=0.5)
    model_name = 'gemini_25_pro'  # deepseek_v3, deepseek, gemini_20, gpt-4o, claude_37_sonnet,  gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
    ablation = ''  # '', 'sat'
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    # append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
    if append == '':
        plt.axvspan(3.2, 7.9, color='gray', alpha=0.5)
    try:
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}_mc.log', 'r').readlines()
    except FileNotFoundError:
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}.log', 'r').readlines()

    model_name = map2title[model_name]
    ablation = 'menu' if ablation == '' else ablation
    plot_path = f'plots/{model_name}/{ablation}{few_shot}/'
    os.makedirs(plot_path, exist_ok=True)
    correct = 0
    dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                      'pred_is_sat': []}
    # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
    #     dataframe_dict['model_count'] = []
    #     dataframe_dict['satisfiability_ratio'] = []

    if model_name == 'GPT-4o' and ablation == 'menu' and few_shot != '_3shot':
        file_lines = file_lines[1000:]

    for id, line in enumerate(file_lines):
        print(id)
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

        # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
        #     continue

        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
        # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
        #     dataframe_dict['model_count'].append(sample_dict['model_count'])
        #     dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)

        if ablation == 'menu':
            try:
                orderable, not_orderable = \
                    parse_generated_output(raw_output=sample_dict['gpt_out'])
            except AttributeError:
                dataframe_dict['num_variables'].pop()
                dataframe_dict['num_clauses'].pop()
                dataframe_dict['is_sat'].pop()
                dataframe_dict['num_prompt_tokens'].pop()
                dataframe_dict['num_completion_tokens'].pop()
                dataframe_dict['alpha'].pop()
                print("Discarding ...")
                continue
                # orderable, not_orderable = None, None

            if not sample_dict['is_sat']:  # if unsat
                if orderable is None and not_orderable is None:  # inconclusive output/solution
                    dataframe_dict['correct'].append(False)
                    dataframe_dict['pred_is_sat'].append(np.nan)
                elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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
            try:
                assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
            except TypeError:
                assignment = None
            if assignment is None:
                dataframe_dict['num_variables'].pop()
                dataframe_dict['num_clauses'].pop()
                dataframe_dict['is_sat'].pop()
                dataframe_dict['num_prompt_tokens'].pop()
                dataframe_dict['num_completion_tokens'].pop()
                dataframe_dict['alpha'].pop()
                print("Discarding ...")
                continue
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

    # Creating the DataFrame
    df = pd.DataFrame(dataframe_dict)
    df[df['alpha'] <= 11]
    few_shot_title = {'': '0-shot', '_3shot': '3-shot'}
    pred_df = df.dropna()
    pred_df['pred_correct'] = ~(pred_df['pred_is_sat'] ^ pred_df['is_sat'])
    accuracy_pred_df = pred_df.groupby('alpha')['pred_correct'].mean().reset_index(name='accuracy')
    window_size = 4
    # accuracy_pred_df['std'] = pred_df.groupby('alpha')['pred_correct'].std().tolist()
    # accuracy_pred_df['min'] = pred_df.groupby('alpha')['pred_correct'].min().tolist()
    # accuracy_pred_df['max'] = pred_df.groupby('alpha')['pred_correct'].max().tolist()
    # accuracy_pred_df['rolling_std'] = accuracy_pred_df['std'].rolling(window=window_size).mean().tolist()
    # accuracy_pred_df['rolling_min'] = accuracy_pred_df['accuracy'].rolling(window=window_size).min().tolist()
    # accuracy_pred_df['rolling_max'] = accuracy_pred_df['accuracy'].rolling(window=window_size).max().tolist()
    accuracy_pred_df['rolling_mean'] = accuracy_pred_df['accuracy'].rolling(window=window_size).mean().tolist()
    accuracy_pred_df_sliced = accuracy_pred_df.iloc[window_size - 1:]
    ax = accuracy_pred_df_sliced.plot(ax=ax, x='alpha',
                                      y='rolling_mean',
                                      kind='line', label='SAT Decision',
                                      linewidth=3)
    # ax.fill_between(accuracy_pred_df_sliced['alpha'],
    #                 accuracy_pred_df_sliced['rolling_max'],
    #                 accuracy_pred_df_sliced['rolling_min'], alpha=0.2)

    # accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
    #     lambda x: np.mean(x['correct'])).reset_index(name='accuracy')
    # window_size = 4
    accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
    window_size = 4  # This is the number of points to include in the moving average
    # acc_mean = accuracy_df_simple['correct'].mean()
    # accuracy_df_simple['acc_mean'] = acc_mean
    # accuracy_df_simple['std'] = df.groupby('alpha')['correct'].std().tolist()
    # accuracy_df_simple['min'] = df.groupby('alpha')['correct'].min().tolist()
    # accuracy_df_simple['max'] = df.groupby('alpha')['correct'].max().tolist()
    # accuracy_df_simple['rolling_std'] = accuracy_df_simple['std'].rolling(window=window_size).mean().tolist()
    # accuracy_df_simple['rolling_min'] = accuracy_df_simple['accuracy'].rolling(window=window_size).min().tolist()
    # accuracy_df_simple['rolling_max'] = accuracy_df_simple['accuracy'].rolling(window=window_size).max().tolist()
    accuracy_df_simple['rolling_mean'] = accuracy_df_simple['accuracy'].rolling(window=window_size).mean().tolist()

    accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]
    ax = accuracy_df_sliced.plot(ax=ax, x='alpha',
                                 y='rolling_mean',
                                 kind='line', label='SAT Search',
                                 linewidth=3)
    # ax.fill_between(accuracy_df_sliced['alpha'],
    #                 accuracy_df_sliced['rolling_mean'] - accuracy_df_sliced['rolling_std'],
    #                 accuracy_df_sliced['rolling_mean'] + accuracy_df_sliced['rolling_std'], alpha=0.2)
    # ax.fill_between(accuracy_df_sliced['alpha'],
    #                 accuracy_df_sliced['rolling_max'],
    #                 accuracy_df_sliced['rolling_min'], alpha=0.2)
    if append == '_2sat':
        pass
        # plt.axvline(x=1.00, color='lime', linestyle='--', linewidth=4)
    else:
        plt.axvline(x=4.24, color='lime', linestyle='--', linewidth=4)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'$ \alpha $', fontsize=22)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'{model_name} Phase Transitions', fontsize=20)
    # plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/combined_decision_search_alpha{append}.png')


def ablation_combined():
    fig, ax = plt.subplots(figsize=(10, 8))
    for ablation in ['', 'sat', 'translate']:
        # if few_shot == '_3shot':
        #     few_shot = ''
        model_name = 'mixtral'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
        # ablation = ''  # ''
        append = ''  # '', '_2sat'
        # few_shot = ''  # '', '_3shot'
        # append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
        try:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}_mc.log', 'r').readlines()
        except FileNotFoundError:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()

        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}
        # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
        #     dataframe_dict['model_count'] = []
        #     dataframe_dict['satisfiability_ratio'] = []

        for id, line in enumerate(file_lines):
            print(id)
            if id in [3024]:
                continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
            #     dataframe_dict['model_count'].append(sample_dict['model_count'])
            #     dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

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
                    elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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
                                                                   menu_items=sample_dict['menu_items'])
                    # verify isomorphism of generated formula
                    pred_true = list(zip(gen_formula, sample_dict['formula']))
                    formula_comparison = \
                        [True if set(pred_clause) == set(true_clause) else False for pred_clause, true_clause in
                         pred_true]
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

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
        accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
            lambda x: np.mean(x['correct'])).reset_index(name='accuracy')

        # Plotting
        # plt.interactive(False)
        ablation_title = {'menu': 'SAT-Menu', 'sat': 'SAT-CNF', 'translate': 'SAT-Translate'}

        # window_size = 4
        accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
        window_size = 3  # This is the number of points to include in the moving average
        # acc_mean = accuracy_df_simple['correct'].mean()
        # accuracy_df_simple['acc_mean'] = acc_mean
        # accuracy_df_simple['std'] = df.groupby('alpha')['correct'].std().tolist()
        # accuracy_df_simple['min'] = df.groupby('alpha')['correct'].min().tolist()
        # accuracy_df_simple['max'] = df.groupby('alpha')['correct'].max().tolist()
        # accuracy_df_simple['rolling_std'] = accuracy_df_simple['std'].rolling(window=window_size).mean().tolist()
        # accuracy_df_simple['rolling_min'] = accuracy_df_simple['accuracy'].rolling(window=window_size).min().tolist()
        # accuracy_df_simple['rolling_max'] = accuracy_df_simple['accuracy'].rolling(window=window_size).max().tolist()
        accuracy_df_simple['rolling_mean'] = accuracy_df_simple['accuracy'].rolling(window=window_size).mean().tolist()

        accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]
        ax = accuracy_df_sliced.plot(ax=ax, x='alpha',
                                     y='rolling_mean',
                                     kind='line', label=ablation_title[ablation],
                                     linewidth=3)
        # ax.fill_between(accuracy_df_sliced['alpha'],
        #                 accuracy_df_sliced['rolling_mean'] - accuracy_df_sliced['rolling_std'],
        #                 accuracy_df_sliced['rolling_mean'] + accuracy_df_sliced['rolling_std'], alpha=0.2)
        # ax.fill_between(accuracy_df_sliced['alpha'],
        #                 accuracy_df_sliced['rolling_max'],
        #                 accuracy_df_sliced['rolling_min'], alpha=0.2)

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'$ \alpha $', fontsize=20)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'{model_name}', fontsize=22)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/combined_menu_sat.png')


def plot_time_acc():
    dir = '~/PycharmProjects/LLM-SAT'
    df_file = os.path.join(dir, 'sat_solved.csv')
    df = pd.read_csv(df_file)

    df['is_sat'] = df['is_sat'].astype(int)
    df = df.groupby(['alpha'])['is_sat'].sum().reset_index()
    df['is_sat'] = df['is_sat'] / df['is_sat'].max()

    df['time'] = df['time'].astype(float)

    # take the average time
    df1 = df.groupby('alpha')['time'].mean().reset_index()
    df1 = df1[df1['alpha'] <= 11]

    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    fig, ax = plt.subplots(figsize=(10, 8))
    model_name = 'gpt-4'

    all_model_list = ['gpt-3.5', 'gpt-4', 'llama-2-70b', 'text-bison@002', 'gemini-pro', 'mixtral']
    few_shot_title = {'': '0-shot', '_3shot': '3-shot'}
    ablation = 'sat'  # '', 'sat', 'translate'

    try:
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}_mc.log', 'r').readlines()
    except FileNotFoundError:
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()
    model_name = map2title[model_name]
    ablation = 'menu' if ablation == '' else ablation
    plot_path = f'plots/{model_name}/{ablation}/'
    os.makedirs(plot_path, exist_ok=True)
    correct = 0
    dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                      'pred_is_sat': []}

    for id, line in enumerate(file_lines):
        print(id)
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)

        if ablation == 'sat':
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

    # Creating the DataFrame
    df = pd.DataFrame(dataframe_dict)
    # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
    # Plotting
    # plt.interactive(False)
    accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
    window_size = 4  # This is the number of points to include in the moving average
    # acc_mean = accuracy_df_simple['correct'].mean()
    # accuracy_df_simple['acc_mean'] = acc_mean
    # accuracy_df_simple['std'] = df.groupby('alpha')['correct'].std().tolist()
    # accuracy_df_simple['rolling_std'] = accuracy_df_simple['std'].rolling(window=window_size).mean().tolist()
    accuracy_df_simple['rolling_mean'] = accuracy_df_simple['accuracy'].rolling(window=window_size).mean().tolist()
    # accuracy_df_simple['rolling_min'] = accuracy_df_simple['accuracy'].rolling(window=window_size).min().tolist()
    # accuracy_df_simple['rolling_max'] = accuracy_df_simple['accuracy'].rolling(window=window_size).max().tolist()
    accuracy_df_sliced = accuracy_df_simple.iloc[window_size - 1:]
    # ax.fill_between(accuracy_df_sliced['alpha'],
    #                 accuracy_df_sliced['rolling_max'],
    #                 accuracy_df_sliced['rolling_min'], alpha=0.2)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    line1, = df1.plot(ax=ax1, kind='line', x='alpha', y='time',
                      label='Solver (MiniSAT)', color='red', linewidth=3,
                      legend=False).get_lines()
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_xlabel('alpha', fontsize=16)
    ax1.set_ylabel('time', fontsize=16)
    ax1.set_yticks(np.arange(0, max(df1['time']) * 1.1, 0.003))
    # ax1.set_xlim([1, 11.5])
    ax1.set_ylim([0, max(df1['time']) * 1.1])
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    # Creating a second y-axis for the second dataframe
    ax2 = ax1.twinx()
    line2, = accuracy_df_sliced.plot(ax=ax2, x='alpha',
                                     y='rolling_mean', color='blue',
                                     kind='line', label=model_name, linewidth=3,
                                     legend=False).get_lines()
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    # ax2.set_xlabel('alpha', fontsize=16)
    ax2.set_ylabel('accuracy', fontsize=16)
    # ax2.set_xticks(fontsize=13)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylim([0, max(accuracy_df_sliced['rolling_mean']) * 1.1])

    plt.title(f'Phase Transition Characteristics', fontsize=18)
    # plt.grid(True)
    plt.tight_layout()
    plt.legend(handles=[line1, line2], fontsize=16)
    # plt.show()
    plt.savefig(f'{plot_path}/gpt_solver_phase_transitions.png')


def plot_model_count_alpha():
    model_name = 'deepseek_v3'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
    ablation = ''  # 'sat'
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    # append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
    file_lines = open(f'../output_mc/{model_name}{ablation}/data_log{append}{few_shot}_mc.log', 'r').readlines()

    model_name = map2title[model_name]
    ablation = 'menu' if ablation == '' else ablation
    plot_path = f'plots/{model_name}/{ablation}/'
    os.makedirs(plot_path, exist_ok=True)
    correct = 0
    dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                      'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                      'pred_is_sat': []}
    # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
    dataframe_dict['model_count'] = []
    dataframe_dict['satisfiability_ratio'] = []

    if model_name == 'GPT-4o' and ablation == 'menu' and few_shot != '_3shot':
        file_lines = file_lines[1000:]

    for id, line in enumerate(file_lines):
        print(id)
        sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

        dataframe_dict['num_variables'].append(sample_dict['num_vars'])
        dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
        dataframe_dict['is_sat'].append(sample_dict['is_sat'])
        dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
        dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
        # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
        dataframe_dict['model_count'].append(sample_dict['model_count'])
        dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

        alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
        dataframe_dict['alpha'].append(alpha)

    # Creating the DataFrame
    df = pd.DataFrame(dataframe_dict)
    df = df[df['is_sat'] == True]
    model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 20)
    easy_df = model_count_df[(model_count_df['alpha'] <= 3.2) | (model_count_df['alpha'] >= 7.9)].groupby(
        ['satisfiability_ratio'])['correct'].mean().reset_index(name='accuracy')
    easy_df['rolling_avg_accuracy'] = easy_df['accuracy'].rolling(window=4).mean().fillna(method='bfill')
    hard_df = model_count_df[(model_count_df['alpha'] < 7.9) & (model_count_df['alpha'] > 3.2)].groupby(
        ['satisfiability_ratio'])['correct'].mean().reset_index(name='accuracy')
    hard_df['rolling_avg_accuracy'] = hard_df['accuracy'].rolling(window=6).mean().fillna(method='bfill')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = easy_df.plot(ax=ax, x='satisfiability_ratio',
                      y='rolling_avg_accuracy',
                      kind='line', label='easy',
                      linewidth=3)
    ax = hard_df.plot(ax=ax, x='satisfiability_ratio',
                      y='rolling_avg_accuracy',
                      kind='line', label='hard',
                      linewidth=3)

    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('satisfiability ratio', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'{model_name}', fontsize=22)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{plot_path}/model_count_alpha.png')


def plot_prob_time():
    dir = '~/PycharmProjects/LLM-SAT'
    df_file = os.path.join(dir, 'sat_solved.csv')
    df = pd.read_csv(df_file)
    df['is_sat'] = df['is_sat'].astype(int)

    # time_df = df.groupby(['alpha'])['is_sat'].sum().reset_index()
    # time_df['is_sat'] = time_df['is_sat'] / df['is_sat'].max()
    time_df = df[df['alpha'] <= 11]
    time_df['time'] = time_df['time'].astype(float)
    # take the average time
    df1 = time_df.groupby('alpha')['time'].mean().reset_index()
    # df1 = df1[df1['alpha'] <= 11]

    prob_df = df[df['alpha'] <= 11]
    prob_df['is_sat'] = prob_df['is_sat'].astype(int)
    df2 = prob_df.groupby(['alpha'])['is_sat'].sum().reset_index()
    df2['is_sat'] = df2['is_sat'] / df2['is_sat'].max()

    fig, ax1 = plt.subplots(figsize=(10, 8))
    plt.axvspan(3.2, 7.9, color='gray', alpha=0.5)
    # plt.axvline(4.2, color='green', linestyle='--')

    line1, = df2.plot(ax=ax1, style='--', x='alpha', y='is_sat',
                      label='Prob(satisfiability)', color='red', linewidth=3,
                      legend=False).get_lines()
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_xlabel(r'$ \alpha $', fontsize=22)
    ax1.set_ylabel('probability', fontsize=18)
    # ax1.set_yticks(np.arange(0, max(df1['time']) * 1.1, 0.003))
    # ax1.set_xlim([1, 11.5])
    # ax1.set_ylim([0, max(df2['is_sat']) * 1.1])
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    # Creating a second y-axis for the second dataframe
    ax2 = ax1.twinx()
    line2, = df1.plot(ax=ax2, x='alpha',
                      y='time', color='blue',
                      kind='line', label='Solver running time',
                      linewidth=3,
                      legend=False).get_lines()
    # ax2.set_yticks(np.arange(0, 1.1, 0.1))
    # ax2.set_xlabel('alpha', fontsize=16)
    ax2.set_ylabel('time', fontsize=18)
    # ax2.set_xticks(fontsize=13)
    ax2.tick_params(axis='y', labelsize=16)
    # ax2.set_ylim([0, max(accuracy_df_sliced['rolling_mean']) * 1.1])
    plt.axvline(x=4.24, color='lime', linestyle='--', linewidth=4)
    plt.title(f'Phase Transitions', fontsize=20)
    # plt.grid(True)
    plt.tight_layout()
    plt.legend(handles=[line1, line2], fontsize=18)
    # plt.show()
    plt.savefig(f'plots/phase_transitions.png')


def random_assign_satisfiability_ratio():
    fig, ax = plt.subplots(figsize=(10, 8))
    model_name = 'gpt-4'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
    ablation = 'sat'  # ''
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    # append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
    try:
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}_mc.log', 'r').readlines()
    except FileNotFoundError:
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}.log', 'r').readlines()

    model_name = map2title[model_name]
    ablation = 'menu' if ablation == '' else ablation
    plot_path = f'plots/{model_name}/{ablation}/'
    os.makedirs(plot_path, exist_ok=True)

    random_flag = False
    for index in range(2):
        if index == 1:
            random_flag = True
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}
        # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
        dataframe_dict['model_count'] = []
        dataframe_dict['satisfiability_ratio'] = []

        for id, line in enumerate(file_lines):
            print(id)
            # if few_shot == '' and id in [3024]:
            #     continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            # if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
            dataframe_dict['model_count'].append(sample_dict['model_count'])
            dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

            alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
            dataframe_dict['alpha'].append(alpha)

            # elif ablation == 'sat':
            if random_flag is False:
                assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
            else:
                if random.random() >= 0.5:
                    assignment = [x for x in range(1, dataframe_dict['num_variables'][0] + 1) if random.random() <= 0.5]
                else:
                    assignment = []
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
        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        # df = df[df['is_sat'] == True]
        # model_count_df = df.groupby('model_count').filter(lambda x: len(x) >= 20)
        accuracy_df = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
        # model_count_df = model_count_df.groupby('satisfiability_ratio')['correct'].mean().reset_index(name='accuracy')
        # model_count_df['rolling_median'] = model_count_df['accuracy'].rolling(window=3).median()
        ax = accuracy_df.plot(ax=ax, x='alpha',
                              y='accuracy',
                              kind='line', label='GPT-4' if not random_flag else 'Random',
                              linewidth=3)
    # ax = hard_df.plot(ax=ax, x='satisfiability_ratio',
    #                   y='rolling_avg_accuracy',
    #                   kind='line', label='hard',
    #                   linewidth=3)

    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.xlabel('satisfiability ratio', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'{model_name}', fontsize=22)
    # plt.grid(True)
    plt.legend(fontsize=22)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/random_assign_satisfiability.png')


def plot_confusion_matrix():
    append = ''  # '', '_2sat'
    few_shot = ''  # '', '_3shot'
    fig, ax = plt.subplots(figsize=(10, 8))

    all_model_list = ['claude_37_sonnet', 'gpt-4o', 'gemini_20', 'deepseek_v3', 'deepseek']

    colors = {
        'Claude 3.7 Sonnet': '#FF5733',  # Coral orange
        'GPT-4o': '#4A90E2',  # Bright blue
        'Gemini 2.0 Flash': '#50C878',  # Emerald green
        'DeepSeek V3': '#9B59B6',  # Royal purple
        'DeepSeek R1': '#2C3E50'  # Dark slate
    }

    for model_name in all_model_list:
        # model_name = 'llama-2-70b'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro, mixtral
        ablation = ''  # '', 'sat', 'translate'
        try:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}_mc.log', 'r').readlines()
        except FileNotFoundError:
            file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}.log', 'r').readlines()
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}

        if model_name == 'GPT-4o' and ablation == 'menu' and append != '_2sat':
            file_lines = file_lines[1000:]

        for id, line in enumerate(file_lines):
            if model_name == 'Mixtral' and id == 3024:
                continue
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
            dataframe_dict['alpha'].append(alpha)

            if ablation == 'menu':
                try:
                    orderable, not_orderable = \
                        parse_generated_output(raw_output=sample_dict['gpt_out'])
                except AttributeError:
                    dataframe_dict['num_variables'].pop()
                    dataframe_dict['num_clauses'].pop()
                    dataframe_dict['is_sat'].pop()
                    dataframe_dict['num_prompt_tokens'].pop()
                    dataframe_dict['num_completion_tokens'].pop()
                    dataframe_dict['alpha'].pop()
                    print(model_name, "Discarding ...")
                    continue

                if not sample_dict['is_sat']:  # if unsat
                    if orderable is None and not_orderable is None:  # inconclusive output/solution
                        dataframe_dict['correct'].append(False)
                        dataframe_dict['pred_is_sat'].append(np.nan)
                    elif len(orderable) == 0 and len(not_orderable) == 0:  # gpt predicts unsat
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

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
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
        plt.show()
        # plt.savefig(f'{plot_path}/cf{append}{few_shot}.png')


if __name__ == '__main__':
    acc_combined()
    # plot_model_count_alpha()
    # output_tokens_acc_corr()
    # decision_search_combined()
    # output_input_token_corr()
    # decision_search_combined()
    # model_count_alpha_combined()
    # unsat_combined()
    # plot_confusion_matrix()
