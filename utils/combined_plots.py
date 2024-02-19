import ast
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_analysis import parse_generated_output, find_best_match
from verify_solution import verify_solution

if __name__ == "__main__":
    plt.figure(figsize=(10, 6))

    for few_shot in ['', '_3shot']:
        model_name = 'gemini-pro'  # gpt-3.5, gpt-4, llama-2-70b, text-bison@002, gemini-pro
        ablation = ''  # '', 'sat', 'translate'
        append = ''  # '', '_2sat'
        # few_shot = ''  # '', '_3shot'
        append_x = '_mc' if model_name == 'gpt-4' and few_shot == '' else ''
        file_lines = open(f'../out_data/{model_name}{ablation}/data_log{append}{few_shot}{append_x}.log', 'r').readlines()

        map2title = {'gpt-4': 'GPT-4', 'gpt-3.5': 'GPT-3.5', 'llama-2-70b': 'Llama-2-70B',
                     'text-bison@002': 'PaLM 2 (text-bison)', 'gemini-pro': 'Gemini Pro',
                     'llama-2-13b': 'Llama-2-13B'}
        model_name = map2title[model_name]
        ablation = 'menu' if ablation == '' else ablation
        plot_path = f'plots/{model_name}/{ablation}/'
        os.makedirs(plot_path, exist_ok=True)
        correct = 0
        dataframe_dict = {'num_variables': [], 'num_clauses': [], 'alpha': [], 'is_sat': [],
                          'correct': [], 'num_prompt_tokens': [], 'num_completion_tokens': [],
                          'pred_is_sat': []}
        if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
            dataframe_dict['model_count'] = []
            dataframe_dict['satisfiability_ratio'] = []

        for id, line in enumerate(file_lines):
            sample_dict = ast.literal_eval(line.split("Data Sample: ")[1])

            # if not sample_dict['is_sat']:   # skip all unsat samples / plot only sat samples
            #     continue

            dataframe_dict['num_variables'].append(sample_dict['num_vars'])
            dataframe_dict['num_clauses'].append(sample_dict['num_clauses'])
            dataframe_dict['is_sat'].append(sample_dict['is_sat'])
            dataframe_dict['num_prompt_tokens'].append(sample_dict['num_prompt_tokens'])
            dataframe_dict['num_completion_tokens'].append(sample_dict['num_completion_tokens'])
            if append != '_2sat' and ablation == 'menu' and few_shot == '' and model_name == 'GPT-4':
                dataframe_dict['model_count'].append(sample_dict['model_count'])
                dataframe_dict['satisfiability_ratio'].append(sample_dict['model_count'] / 2 ** sample_dict['num_vars'])

            alpha = sample_dict['num_clauses'] / sample_dict['num_vars']
            dataframe_dict['alpha'].append(alpha)

            if ablation == 'menu':
                try:
                    orderable, not_orderable = \
                        parse_generated_output(raw_output=sample_dict['gpt_out'])
                except AttributeError:
                    orderable, not_orderable = [], []

                if not sample_dict['is_sat']:  # if unsat
                    if len(orderable) == 0 and len(not_orderable) == 0 \
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
                    if len(orderable) == 0 and len(not_orderable) == 0:
                        dataframe_dict['pred_is_sat'].append(False)  # pred unsat
                    else:
                        dataframe_dict['pred_is_sat'].append(True)  # pred unsat

            elif ablation == 'sat':
                assignment = parse_generated_output_sat(raw_output=sample_dict['gpt_out'])
                # only use those variables in the assignment which are part of the formula
                # sometimes the LLM creates and assigns values to new variables
                assignment = list(set(assignment).intersection(np.arange(1, sample_dict['num_vars'] + 1, 1)))
                if not sample_dict['is_sat']:  # if unsat
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
                item_nums = np.arange(1, len(menu_items) + 1)
                item2num = dict(zip(menu_items, item_nums))
                try:
                    gen_formula = parse_generated_output_translate(raw_output=sample_dict['gpt_out'],
                                                                   item_to_number=item2num)
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

            else:
                raise NotImplemented

        # Creating the DataFrame
        df = pd.DataFrame(dataframe_dict)
        # Grouping by 'num_variables' and then calculating accuracy for each 'alpha' within those groups
        accuracy_df = df.groupby(['num_variables', 'alpha']).apply(
            lambda x: np.mean(x['correct'])).reset_index(name='accuracy')

        # Plotting
        # plt.interactive(False)
        few_shot_title = {'': '0-shot', '_3shot': '3-shot'}
        # plt.figure(figsize=(10, 6))
        # for num_vars in accuracy_df['num_variables'].unique():
        #     # if num_vars in [4,6,10]:
        #         subset = accuracy_df[accuracy_df['num_variables'] == num_vars]
        #         plt.plot(subset['alpha'], subset['accuracy'], marker='o', label=f'Variables: {num_vars}')
        #
        # plt.xlabel('alpha', fontsize=16)
        # plt.ylabel('accuracy', fontsize=16)
        # plt.yticks(np.arange(0, 1.1, 0.1), fontsize=13)
        # plt.xticks(fontsize=13)
        # plt.legend()
        # plt.title(f'{model_name} ({few_shot_title[few_shot]})', fontsize=18)
        # plt.grid(True)
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig(f'{plot_path}/all_alpha{append}{few_shot}.png')

        accuracy_df_simple = df.groupby('alpha')['correct'].mean().reset_index(name='accuracy')
        plt.plot(accuracy_df_simple['alpha'], accuracy_df_simple['accuracy'],
                 marker='o', label=few_shot_title[few_shot])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('alpha', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f'{model_name}', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{plot_path}/combined_mean_alpha{append}.png')
