import os
import matplotlib.pyplot as plt
from collections import Counter


# Configure logging
class DataLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, data_sample):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write(str(data_sample) + '\n')


# A function to log data from each data sample
def log_data_sample(model_name, num_vars, num_clauses, formula, is_sat, preferences, menu_items,
                    gpt_out, num_prompt_tokens, num_completion_tokens, ablation='', job_num='', two_sat_flag=False):
    ablation = '' if ablation == 'menu' else ablation
    if ablation == 'sat':
        menu_items = ''
        preferences = ''
    out_data_path = f'out_data/{model_name}{ablation}'
    if not os.path.exists(out_data_path):
        os.makedirs(out_data_path)

    append = '_2sat' if two_sat_flag else ''
    # logger = DataLogger(os.path.join(out_data_path, f'data_log{append}.log'))
    data_log_file_name = f'data_log{append}.log'
    if job_num != '':
        data_log_file_name = f'job{job_num}_{data_log_file_name}'
    logger = DataLogger(os.path.join(out_data_path, data_log_file_name))

    data_sample = {
        'num_vars': num_vars,
        'num_clauses': num_clauses,
        'formula': formula,
        'is_sat': is_sat,
        'preferences': preferences,
        'menu_items': menu_items,
        'gpt_out': gpt_out,
        'num_prompt_tokens': num_prompt_tokens,
        'num_completion_tokens': num_completion_tokens
    }
    logger.log(f"Data Sample: {data_sample}")

# A function to plot the distribution of numeric data
def plot_distribution(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=20, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# A function to plot comparison of two variables
def plot_comparison(x_data, y_data, x_label, y_label, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, alpha=0.7, color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_bar_comparison(x_data, y_data, x_label, y_label, title):
    plt.figure(figsize=(10, 5))
    combined = list(zip(x_data, y_data))
    # Count the number of True values for each unique element in the values list
    true_counts = Counter(value for value, flag in combined if flag)

    # Prepare data for plotting
    unique_values = list(set(x_data))
    counts = [true_counts.get(value, 0) for value in unique_values]

    # Plotting
    plt.bar(unique_values, counts)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_dicts(my_dict, x_label, y_label, title):
    lists = sorted(my_dict.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
