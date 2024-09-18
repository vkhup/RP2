import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the group performance summary data
file_path = '/statistics/data_output/group_comparison_results.csv'
group_data = pd.read_csv(file_path)

# Extract the prompt type (e.g., SSP, CEP, CT) from the 'Group' column
group_data['Prompt_Type'] = group_data['Group'].str.replace('_novel', '').str.replace('_base', '')

# Function to calculate Cohen's d (effect size)
def calculate_cohens_d(row_novel, row_base):
    # Pooled standard deviation
    pooled_sd = ((row_novel['Std'] ** 2 * (10 - 1)) + (row_base['Std'] ** 2 * (10 - 1))) / (10 + 10 - 2)
    pooled_sd = pooled_sd ** 0.5

    # Cohen's d
    cohen_d = (row_novel['Mean'] - row_base['Mean']) / pooled_sd
    return cohen_d

# Define the metrics and initialize effect sizes
effect_sizes = {
    'bleu_scores': [],
    'rouge_l_scores': [],
    'rouge_1_scores': [],
    'rouge_2_scores': [],
    'context_similarity_scores': []
}
# List of unique prompt types
prompt_types = group_data['Prompt_Type'].unique()

# Calculate Cohen's d for each prompt type (handling data directly from 'Novel Mean' and 'Base Mean')
for prompt_type in prompt_types:
    for metric in effect_sizes.keys():
        data = group_data[(group_data['Prompt_Type'] == prompt_type) & (group_data['Metric'] == metric)]

        # If no data, append NaN to maintain shape consistency
        if data.empty:
            print(f"Missing data for {prompt_type} in {metric}. Adding NaN.")
            effect_sizes[metric].append(np.nan)
            continue

        # Extract relevant values for novel and base
        novel_mean = data['Novel Mean'].values[0]
        base_mean = data['Base Mean'].values[0]
        novel_std = data['Novel Std Dev'].values[0]
        base_std = data['Base Std Dev'].values[0]

        # Calculate Cohen's d and append to effect_sizes
        effect_sizes[metric].append(calculate_cohens_d(
            row_novel={'Mean': novel_mean, 'Std': novel_std},
            row_base={'Mean': base_mean, 'Std': base_std}
        ))

# Function to plot effect sizes with custom subplot titles and figure title
def plot_effect_size_with_custom_titles(effect_sizes, metrics, prompt_types, custom_titles, fig_title, n_rows, n_cols,
                                        start_label_index):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows), dpi=300)  # Increase figure size based on rows
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]  # Flatten axes if multiple, otherwise make a list

    # Create labels starting from the provided index (A, B, C...)
    subplot_labels = [chr(i) for i in range(65 + start_label_index, 65 + start_label_index + len(metrics))]

    # Loop through each metric and create a subplot for each
    for i, (metric_name, custom_title) in enumerate(zip(metrics, custom_titles)):
        ax = axes[i]  # Current axis
        ax.set_title(f"{subplot_labels[i]}. {custom_title}", loc='left')  # Add subplot label (A, B, C...)

        # Plot the effect sizes for the current metric
        bars = ax.bar(prompt_types, effect_sizes[metric_name], color='skyblue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Prompt Type")

        # Annotate the bars with Cohen's d values dynamically above or below the bars
        for bar in bars:
            yval = bar.get_height()
            if np.isfinite(yval):  # Only annotate if the value is valid
                if yval > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, yval - 0.02, f'{yval:.2f}', ha='center', va='top')
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        ax.tick_params(axis='x', rotation=45)

    axes[0].set_ylabel("Cohen's d")
    plt.subplots_adjust(top=0.95)  # Adjust this value to control the title space
    fig.suptitle(fig_title, fontsize=16, y=0.98)
    fig.savefig(f"{fig_title}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

metrics_1 = ['bleu_scores', 'rouge_1_scores', 'rouge_2_scores']
custom_titles_1 = ['BLEU Scores Effect Size (Cohen\'s d)', 'ROUGE-1 Scores Effect Size (Cohen\'s d)', 'ROUGE-2 Scores Effect Size (Cohen\'s d)']

plot_effect_size_with_custom_titles(effect_sizes, metrics_1, prompt_types, custom_titles_1, fig_title="Effect Size Comparison Across Group Categories (Part 1)",  n_rows=3, n_cols=1, start_label_index=0)

metrics_2 = ['rouge_l_scores', 'context_similarity_scores']
custom_titles_2 = ['ROUGE-L Scores Effect Size (Cohen\'s d)', 'Context Similarity Effect Size (Cohen\'s d)']

plot_effect_size_with_custom_titles(effect_sizes, metrics_2, prompt_types,  custom_titles_2, fig_title="Effect Size Comparison Across Group Categories (Part 2)", n_rows=2, n_cols=1, start_label_index=3)
