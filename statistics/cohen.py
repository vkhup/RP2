import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
file_path_stats = '/statistics/data_output/individual_prompt_statistics.csv'
file_path_tests = '/statistics/data_output/individual_p_values.csv'

data_stats = pd.read_csv(file_path_stats)
data_tests = pd.read_csv(file_path_tests)

# Filter data_tests to only include rows where 'Significant' is True
data_tests_significant = data_tests[data_tests['Significant'] == True][['Prompt ID', 'Metric', 'Test Used']]

# Separate the Novel and Base data based on the Condition column
novel_data = data_stats[data_stats['Condition'] == 'Novel']
base_data = data_stats[data_stats['Condition'] == 'Base']

# Merge the Novel and Base data on 'Prompt ID' and 'Metric'
merged_data = pd.merge(novel_data, base_data, on=['Prompt ID', 'Metric'], suffixes=('_novel', '_base'))

# Merge the test data (from data_tests_significant) with the merged novel and base data
data_merged = pd.merge(merged_data, data_tests_significant, on=['Prompt ID', 'Metric'], how='inner')  # 'inner' to keep only significant results

# Add a column for the type of effect size calculation
def calculate_effect_size(row):
    if row['Test Used'] == 't-test':
        # Cohen's d calculation for parametric data
        pooled_sd = ((row['Std Dev_novel'] ** 2 * (10 - 1)) + (row['Std Dev_base'] ** 2 * (10 - 1))) / (10 + 10 - 2)
        pooled_sd = pooled_sd ** 0.5
        cohen_d = (row['Mean_novel'] - row['Mean_base']) / pooled_sd
        return cohen_d
    elif row['Test Used'] == 'Wilcoxon':
        # Cliff's delta calculation for non-parametric data
        cliff_delta = (row['Median_novel'] - row['Median_base']) / (row['Max_novel'] - row['Min_base'])
        return cliff_delta
    else:
        return None

# Apply the effect size calculation based on the test used
data_merged['Effect_Size'] = data_merged.apply(calculate_effect_size, axis=1)

# Create a single column for the effect size (without suffixes in the plot)
effect_size_data = data_merged[['Prompt ID', 'Metric', 'Effect_Size', 'Test Used']]

# Separate the parametric and non-parametric data
parametric_data = effect_size_data[effect_size_data['Test Used'] == 't-test']
non_parametric_data = effect_size_data[effect_size_data['Test Used'] == 'Wilcoxon']

# Print available metrics to debug
print(f"Available metrics for parametric data: {parametric_data['Metric'].unique()}")
print(f"Available metrics for non-parametric data: {non_parametric_data['Metric'].unique()}")

# Update metrics_to_plot to reflect actual metric names in the dataset
metrics_to_plot = ['bleu_score', 'rouge_1', 'rouge_2', 'rouge_L', 'context_similarity']

# Function to plot effect sizes for parametric or non-parametric data
def plot_effect_sizes(data, metrics, title, fig_name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        metric_data = data[data['Metric'] == metric]
        if metric_data.empty:
            print(f"No data to plot for {metric}")
            continue  # Skip this metric if no data

        print(f"Data found for {metric}: {len(metric_data)} rows")  # Debugging: print the number of rows
        ax = axes[i]
        ax.bar(metric_data['Prompt ID'], metric_data['Effect_Size'], color='skyblue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Prompt ID")
        ax.set_ylabel("Effect Size")
        ax.set_title(f"Effect Size for {metric}", fontsize=14)
        ax.tick_params(axis='x', rotation=45)

        # Annotate the bars with effect size values
        for bar in ax.patches:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, yval - 0.02, f'{yval:.2f}', ha='center', va='top')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

    # Remove any extra axes if metrics are less than 6
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{fig_name}.png', format='png', dpi=300)
    plt.show()

# Plot parametric data in 2 figures (each with a 2x3 layout)
plot_effect_sizes(parametric_data, metrics_to_plot[:3], title="Effect Sizes for Parametric Data (Part 1)", fig_name="parametric_part_1")
plot_effect_sizes(parametric_data, metrics_to_plot[3:], title="Effect Sizes for Parametric Data (Part 2)", fig_name="parametric_part_2")

# Plot non-parametric data in 2 figures (each with a 2x3 layout)
plot_effect_sizes(non_parametric_data, metrics_to_plot[:3], title="Effect Sizes for Non-Parametric Data (Part 1)", fig_name="non_parametric_part_1")
plot_effect_sizes(non_parametric_data, metrics_to_plot[3:], title="Effect Sizes for Non-Parametric Data (Part 2)", fig_name="non_parametric_part_2")
