import os
import matplotlib.pyplot as plt
from collections import Counter

out_data_path = 'out_data'
if not os.path.exists(out_data_path):
    os.makedirs(out_data_path)
# Configure logging
class DataLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, data_sample):
        with open(self.log_file, 'a') as file:
            file.write(str(data_sample) + '\n')


logger = DataLogger(os.path.join(out_data_path, 'data_log.log'))
# A function to log data from each data sample
def log_data_sample(num_vars, num_clauses, formula, is_sat, preferences, menu_items,
                    gpt_out, num_prompt_tokens, num_completion_tokens):
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

# Example of how to use these functions with sample data
# You would replace these with the actual data collection process

# Assume we have a list of data samples in the following format
# data_samples = [
#     # num_vars, num_classes, formula, is_sat, preferences, menu_items, gpt_out, num_prompt_tokens, num_completion_tokens
#     (3, 2, [[1, -2], [-1, 2, 3]], True, 'high', 'burger', 'GPT output 1', 100, 150),
#     # Add more data samples as needed
# ]

# # Log and plot data
# for sample in data_samples:
#     log_data_sample(*sample)

# Extract specific fields for plotting
# num_vars_list = [sample[0] for sample in data_samples]
# num_classes_list = [sample[1] for sample in data_samples]
# is_sat_list = [sample[3] for sample in data_samples]
# num_prompt_tokens_list = [sample[7] for sample in data_samples]
# num_completion_tokens_list = [sample[8] for sample in data_samples]
#
# # Plot distributions
# plot_distribution(num_vars_list, 'Distribution of num_vars', 'num_vars', 'Frequency')
# plot_distribution(num_classes_list, 'Distribution of num_classes', 'num_classes', 'Frequency')
# plot_distribution(is_sat_list, 'Distribution of is_sat', 'is_sat', 'Frequency')
# plot_distribution(num_prompt_tokens_list, 'Distribution of Prompt Tokens', 'num_prompt_tokens', 'Frequency')
# plot_distribution(num_completion_tokens_list, 'Distribution of Completion Tokens', 'num_completion_tokens', 'Frequency')
#
# # Plot comparisons
# plot_comparison(num_vars_list, num_prompt_tokens_list, 'num_vars', 'num_prompt_tokens', 'Comparison of num_vars and num_prompt_tokens')
# plot_comparison(num_classes_list, num_completion_tokens_list, 'num_classes', 'num_completion_tokens', 'Comparison of num_classes and num_completion_tokens')