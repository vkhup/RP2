import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/statistics/data_output/cumulative_novel_vs_base_statistical_results.csv'
data_significant_prompts = pd.read_csv(file_path)

# Extract the prompt type (e.g., SSP, CEP, EXP) from the 'Group 1' column
data_significant_prompts['Prompt_Type'] = data_significant_prompts['Group'].str.extract(r'(\D+)', expand=False)

# Clean up 'Group 1' to get the base prompt names (e.g., SS5 instead of SS5_novel)
data_significant_prompts['Prompt'] = data_significant_prompts['Group'].str.replace('_novel', '').str.replace('_base', '')

# Function to calculate Cohen's d (effect size)
def calculate_cohens_d(row):
    # Pooled standard deviation
    pooled_sd = ((row['Novel Std Dev'] ** 2 * (10 - 1)) + (row['Base Std Dev'] ** 2 * (10 - 1))) / (10 + 10 - 2)
    pooled_sd = pooled_sd ** 0.5

    # Cohen's d
    cohen_d = (row['Novel Mean'] - row['Base Mean']) / pooled_sd
    return cohen_d


# Apply the Cohen's d calculation to each row
data_significant_prompts['Cohens_d'] = data_significant_prompts.apply(calculate_cohens_d, axis=1)

# Function to plot effect sizes with custom subplot titles and figure title
def plot_effect_size_with_custom_titles(data, metrics, custom_titles, fig_title, n_rows, n_cols, start_label_index):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows), dpi=300)  # Increase figure size based on rows
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]  # Flatten axes if multiple, otherwise make a list

    # Create labels starting from the provided index (A, B, C...)
    subplot_labels = [chr(i) for i in range(65 + start_label_index, 65 + start_label_index + len(metrics))]

    # Loop through each metric and create a subplot for each
    for i, (metric_name, custom_title) in enumerate(metrics):
        ax = axes[i]  # Current axis
        metric_data = data[data['Metric'] == metric_name]
        ax.set_title(f"{subplot_labels[i]}. {custom_title}", loc='left')  # Add subplot label (A, B, C...)

        # Filter data for the current metric and plot effect sizes
        prompt_types = metric_data['Prompt_Type'].unique()

        for prompt_type in prompt_types:
            # Filter data for current prompt type and plot
            prompt_data = metric_data[metric_data['Prompt_Type'] == prompt_type]

            # Plot bar chart for each prompt type
            bars = ax.bar(prompt_data['Prompt'], prompt_data['Cohens_d'], color='skyblue')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_xlabel("Prompt")

            # Annotate the bars with Cohen's d values dynamically above or below the bars
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, yval - 0.02, f'{yval:.2f}', ha='center', va='top')
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

        # Rotate x labels for better readability
        ax.tick_params(axis='x', rotation=45)

    # Set y-axis label on the leftmost plot
    axes[0].set_ylabel("Cohen's d")

    # Adjust layout to avoid overlap and leave space for the figure title
    plt.subplots_adjust(top=0.95)  # Adjust this value to control the title space

    # Add the figure title after layout adjustment
    fig.suptitle(fig_title, fontsize=16, y=0.98)

    fig.savefig(f"{fig_title}.png", format='png', dpi=300, bbox_inches='tight')

    plt.show()


# Define the metrics and their custom subplot titles
metrics_1 = [
    ('bleu_scores', 'BLEU Scores Comparison: Novel vs. Base'),
    ('rouge_1_scores', 'ROUGE-1 Scores Comparison: Novel vs. Base'),
    ('rouge_2_scores', 'ROUGE-2 Scores Comparison: Novel vs. Base')
]

metrics_2 = [
    ('rouge_l_scores', 'ROUGE-L Scores Comparison: Novel vs. Base'),
    ('context_similarity_scores', 'Context Similarity Scores: Novel vs. Base')
]

# Plot the first figure with custom titles and labels (A, B, C)
plot_effect_size_with_custom_titles(data_significant_prompts, metrics_1,   [title for _, title in metrics_1],   fig_title="Effect Size Comparison Metrics Across Individual Prompts  (Part 1)", n_rows=3, n_cols=1, start_label_index=0)

# Plot the second figure with custom titles and labels (D, E)
plot_effect_size_with_custom_titles(data_significant_prompts, metrics_2,  [title for _, title in metrics_2], fig_title="Effect Size Comparison Metrics Across Individual Prompts (Part 2)",  n_rows=2, n_cols=1, start_label_index=3)
