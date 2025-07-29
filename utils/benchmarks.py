import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Data
data = {
    "Dataset": [
        "MATH-500 \n(Pass@1)",
        "MATH",
        "GSM8k",
        "MMLU-Pro",
        "SWE-bench \nVerified",
        "Aider-Polyglot \n(Acc.)",
        "BigCodeBench \n(Hard Set Complete)",
    ],
    "DeepSeek R1": [97.3, 90.45, 96.13, 84.0, 49.2, 53.3, 40.5],
    "Competitor": [88, 85.87, 96.4, 77.9, 62.3, 45.3, 39.2],
    "Competitor Name": [
        "Gemini 2.0 Flash",
        "Gemini 2.0 Flash",
        "Claude 3.5 Sonnet",
        "GPT-4o",
        "Claude 3.7 Sonnet",
        "Claude 3.5 Sonnet",
        "Claude 3.7 Sonnet",
    ],
}

df = pd.DataFrame(data)

# Define color sets
color_set1 = {"DeepSeek R1": "#ffb400", "Competitor": "#f46e27"}
color_set2 = {"DeepSeek R1": "#4A90E2", "Competitor": "#005A9C"}

# Plotting with competitor names overlaid on the bars
fig, ax = plt.subplots(figsize=(18, 10))
bar_width = 0.35
x = range(len(df))

# Bars with specified colors
for i in x:
    if df["Dataset"][i] in ["MATH-500 \n(Pass@1)", "MATH", "GSM8k", "MMLU-Pro"]:
        color_deepseek = color_set1["DeepSeek R1"]
        color_competitor = color_set1["Competitor"]
    else:
        color_deepseek = color_set2["DeepSeek R1"]
        color_competitor = color_set2["Competitor"]

    ax.bar(
        i - bar_width / 2,
        df["DeepSeek R1"][i],
        width=bar_width,
        color=color_deepseek,
        zorder=2,
    )
    ax.bar(
        i + bar_width / 2,
        df["Competitor"][i],
        width=bar_width,
        color=color_competitor,
        zorder=2,
    )

# Create custom legend patches
math_deepseek = mpatches.Patch(
    color=color_set1["DeepSeek R1"], label="DeepSeek R1 (mathematics)"
)
math_competitor = mpatches.Patch(
    color=color_set1["Competitor"], label="Competitor (mathematics)"
)
coding_deepseek = mpatches.Patch(
    color=color_set2["DeepSeek R1"], label="DeepSeek R1 (coding)"
)
coding_competitor = mpatches.Patch(
    color=color_set2["Competitor"], label="Competitor (coding)"
)

# Add legend with custom handles
ax.legend(
    handles=[math_deepseek, math_competitor, coding_deepseek, coding_competitor],
    loc="best",
    fontsize=20,
)

# Labels and aesthetics
# ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Performance", fontsize=25)
ax.set_title("Benchmarks: DeepSeek R1 vs Competitors", fontsize=18, fontweight="bold")
ax.set_xticks(x)
ax.tick_params(axis="y", labelsize=20)  # Set y-tick font size
ax.set_xticklabels(df["Dataset"], rotation=30, ha="right", fontsize=20)

# Annotate competitor names on the bars
for i in x:
    ax.text(
        i + bar_width / 2,
        df["Competitor"][i] / 2,
        df["Competitor Name"][i],
        ha="center",
        va="center",
        fontsize=17,
        rotation=90,
        color="white",
        fontweight="bold",
    )

# Grid and layout
ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=1)
plt.tight_layout()
plt.show()
