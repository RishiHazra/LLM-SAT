import fcntl
import os
import pickle as pkl
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from utils import map2title


# Configure logging
class DataLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, data_sample):
        with open(self.log_file, "a", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(str(data_sample) + "\n")
            fcntl.flock(file, fcntl.LOCK_UN)


# A function to log data from each data sample
def log_data_sample(
    model_name,
    num_vars,
    num_clauses,
    formula,
    is_sat,
    preferences,
    menu_items,
    gpt_out,
    num_prompt_tokens,
    num_completion_tokens,
    ablation="",
    job_num="",
    few_shot=0,
    sat_class=None,
):
    ablation = "" if ablation == "menu" else ablation
    if ablation == "sat":
        menu_items = ""
        preferences = ""
    out_data_path = f"out_data/{sat_class}/{model_name}{ablation}"
    if not os.path.exists(out_data_path):
        os.makedirs(out_data_path)

    append_few_shot = f"_{few_shot}shot" if few_shot != 0 else ""
    data_log_file_name = f"data_log{append_few_shot}.log"
    if job_num != "":
        data_log_file_name = f"job{job_num}_{data_log_file_name}"
    logger = DataLogger(os.path.join(out_data_path, data_log_file_name))

    data_sample = {
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "formula": formula,
        "is_sat": is_sat,
        "preferences": preferences,
        "menu_items": menu_items,
        "gpt_out": gpt_out,
        "num_prompt_tokens": num_prompt_tokens,
        "num_completion_tokens": num_completion_tokens,
    }
    logger.log(f"Data Sample: {data_sample}")


def sample_in_context(few_shot, model_name, ablation):
    model_name = map2title[model_name]
    in_context_dir = f"utils/in_context_examples/{model_name}/{ablation}/examples"
    file_lines = open(in_context_dir).readlines()
    all_examples = []  # file only stores correct examples
    for i in range(0, len(file_lines), 3):
        pair = "\n".join([file_lines[i], file_lines[i + 1]])
        all_examples.append(pair)
    np.random.shuffle(all_examples)
    sampled_examples = all_examples[:few_shot]
    return "\n".join(sampled_examples)


# A function to plot the distribution of numeric data
def plot_distribution(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=20, alpha=0.7, color="blue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# A function to plot comparison of two variables
def plot_comparison(x_data, y_data, x_label, y_label, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, alpha=0.7, color="red")
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


def plot_dataset_analysis():
    def load_and_process(data_path, df_dict):
        sat_data = pkl.load(open(data_path, "rb"))
        for ind, num_vars, num_clauses, formula, is_sat, _ in sat_data:
            df_dict["num_variables"].append(num_vars)
            df_dict["num_clauses"].append(num_clauses)
            df_dict["is_sat"].append(is_sat)
            alpha = num_clauses / num_vars
            dataframe_dict["alpha"].append(alpha)
        return df_dict

    dataframe_dict = {"num_variables": [], "num_clauses": [], "alpha": [], "is_sat": []}
    data_path = os.path.join(os.environ["DATA_PATH"], "dataset_3sat_int.pkl")
    dataframe_dict = load_and_process(data_path, dataframe_dict)
    data_path = os.path.join(os.environ["DATA_PATH"], "dataset_3sat_float.pkl")
    dataframe_dict = load_and_process(data_path, dataframe_dict)
    dataframe = pd.DataFrame(dataframe_dict)
    # figure = plt.figure(figsize=(10,8))
    # Grouping by num_variables and is_sat to get count
    grouped = (
        dataframe.groupby(["num_variables", "is_sat"]).size().unstack(fill_value=0)
    )

    # Preparing data for the pie chart
    sizes = grouped.sum(axis=1).values
    inner_labels = grouped.index.values
    outer_sizes = grouped.values.flatten()
    outer_labels = ["unSAT", "SAT"] * len(grouped)

    # Nice color palette for the inner pie
    # inner_colors = plt.cm.Set2.colors[:len(inner_labels)]
    # Deciding to make the wedge corresponding to '3' variables pop out as an example
    inner_explode = [0.2 if label == 4 else 0 for label in inner_labels]
    # For every 'True' in inner_explode, we need two values in outer_explode (one for SAT and one for UNSAT)
    outer_explode = []
    for e in inner_explode:
        outer_explode.extend(
            [e, e]
        )  # Repeat the explode value for both SAT and UNSAT segments

    # Define colors for SAT (True) and UNSAT (False)
    # inner_colors = plt.cm.Set1.colors[:len(inner_labels)]  # ['lightgray', 'k']
    inner_colors = plt.cm.Set2.colors[: len(inner_labels)]
    colors = plt.cm.Set3.colors[
        :2
    ]  # sns.color_palette("Set2", 2)  # plt.cm.Pastel1.colors[:len(outer_labels)]
    # Manually specifying colors for each segment of the outer pie to ensure consistency
    outer_colors = [colors[i % 2] for i in range(len(outer_labels))]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))

    outer_wedges, _ = ax.pie(
        outer_sizes,
        radius=1.3,
        labeldistance=0.8,
        explode=outer_explode,
        wedgeprops=dict(width=0.4, edgecolor="w"),
        colors=outer_colors,
    )
    inner_pie, _ = ax.pie(
        sizes,
        radius=1,
        colors=inner_colors,
        explode=inner_explode,
        wedgeprops=dict(width=0.6, edgecolor="w"),
    )
    outer_legend_elements = [
        Patch(facecolor=outer_colors[1], edgecolor="w", label="SAT"),
        Patch(facecolor=outer_colors[0], edgecolor="w", label="unSAT"),
    ]

    # Creating legends for both pies
    # inner_legend = ax.legend(inner_pie, inner_labels, title="# Variables",
    #                          bbox_to_anchor=(-0.3, -0.1, 0.5, 1), fontsize=16, title_fontsize=18)
    # outer_legend = ax.legend(handles=outer_legend_elements, title="Satisfiability", loc="center left",
    #                          bbox_to_anchor=(-0.3, 0.5, 0.5, 1), fontsize=16, title_fontsize=18)
    # Adding legends back to the plot (to display both legends simultaneously)
    # ax.add_artist(inner_legend)
    plt.savefig("plots/dataset_pie.png")

    df_sample = dataframe
    df_sample["is_sat"] = df_sample["is_sat"].map({False: "unSAT", True: "SAT"})
    colors = {"SAT": outer_colors[1], "unSAT": outer_colors[0]}
    # Boxplot of num_clauses by is_sat
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="is_sat", y="num_clauses", data=df_sample, palette=colors)
    plt.xlabel("")  # Remove the x-axis label
    plt.ylabel("# clauses", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Distribution of clauses", fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/dataset_boxplot.png")
    # plt.show()

    # Violin Plot of alpha by is_sat
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="is_sat", y="alpha", data=df_sample, palette=colors)
    plt.xlabel("")  # Remove the x-axis label
    plt.ylabel("alpha", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Distribution of alpha", fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/dataset_violinplot.png")

    # Setting up the plot
    plt.figure(figsize=(8, 6))
    # Creating a histogram with a kernel density estimate
    ax = sns.kdeplot(
        data=df_sample,
        x="alpha",
        hue="is_sat",
        common_norm=False,
        palette=colors,
        fill=True,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="black",
    )
    plt.xlabel("")  # Remove the x-axis label
    plt.ylabel("alpha", fontsize=18)  # Setting y-axis label and font size
    plt.xticks(fontsize=18)  # Setting x-axis tick font size
    plt.yticks(fontsize=18)  # Setting y-axis tick font size
    plt.title(
        "Distribution of alpha", fontsize=20
    )  # Setting the title and its font size
    # Get the legend, replace labels
    # handles, labels = ax.get_legend_handles_labels()
    # Assuming 'is_sat' has boolean values, you can customize as needed:
    # new_labels = ['SAT' if label == 'True' else 'unSAT' for label in labels]
    # ax.legend(handles=handles, labels=new_labels, title='is SAT', title_fontsize='13', fontsize='12')
    plt.tight_layout()
    plt.savefig("plots/dataset_distribution_plot.png")
    # plt.show()
