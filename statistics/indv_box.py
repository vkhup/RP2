import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from statsmodels.stats.multitest import multipletests


# Function to load data from JSON and extract relevant metrics
def load_data(file_path, condition, group_name):
    with open(file_path, 'r') as f:
        data = json.load(f)

    extracted_data = []
    for entry in data:
        extracted_data.append({
            'id': entry['id'],
            'prompt': entry['prompt'],
            'bleu_score': entry.get('bleu_score', None),
            'rouge_1': entry.get('rouge_1', None),
            'rouge_2': entry.get('rouge_2', None),
            'rouge_L': entry.get('rouge_L', None),
            'context_similarity': entry.get('context_similarity', None),
            'condition': condition,
            'group': group_name
        })
    return extracted_data


# Function to load and combine novel and base data
def combine_data(group_files):
    combined_data = []
    for group_name, paths in group_files.items():
        novel_data = load_data(paths['novel'], 'Novel', group_name)
        base_data = load_data(paths['base'], 'Base', group_name)
        combined_data.extend(novel_data)
        combined_data.extend(base_data)
    return pd.DataFrame(combined_data)


# Function to calculate and save boxplot statistics
def calculate_statistics(data, metrics):
    stats_list = []

    for metric in metrics:
        for prompt_id in data['id'].unique():
            for condition in data['condition'].unique():
                metric_data = data[(data['metric'] == metric) &
                                   (data['id'] == prompt_id) &
                                   (data['condition'] == condition)]['value']

                if not metric_data.empty:
                    stats = {
                        'Metric': metric,
                        'Prompt ID': prompt_id,
                        'Condition': condition,
                        'Mean': metric_data.mean(),
                        'Median': metric_data.median(),
                        'Std Dev': metric_data.std(),
                        'IQR': metric_data.quantile(0.75) - metric_data.quantile(0.25),
                        'Min': metric_data.min(),
                        'Max': metric_data.max()
                    }
                    stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv('',
                    index=False)
    print("Statistics saved to ''")
    return stats_df


# Function to run statistical tests for each prompt ID and apply multiple testing correction per metric
def perform_statistical_tests(data):
    all_results = []  # Store all results in a list to save in a single CSV file
    all_significant_prompts = {}

    for metric in data['metric'].unique():
        p_values = []
        results = []

        for prompt_id in data['id'].unique():
            # Get Novel and Base scores for the given prompt and metric
            novel_values = data[(data['id'] == prompt_id) & (data['metric'] == metric) & (data['condition'] == 'Novel')]['value'].dropna()
            base_values = data[(data['id'] == prompt_id) & (data['metric'] == metric) & (data['condition'] == 'Base')]['value'].dropna()

            if len(novel_values) == len(base_values) and len(novel_values) > 0:
                # Check normality using Shapiro-Wilk test
                stat_novel, p_novel = shapiro(novel_values)
                stat_base, p_base = shapiro(base_values)

                if p_novel > 0.05 and p_base > 0.05:
                    # If normally distributed, use paired t-test
                    stat, p_val = ttest_rel(novel_values, base_values)
                    test_used = 't-test'
                else:
                    # If not normally distributed, use Wilcoxon signed-rank test
                    stat, p_val = wilcoxon(novel_values, base_values)
                    test_used = 'Wilcoxon'

                # Append p-values for multiple testing correction
                p_values.append(p_val)
                results.append({
                    'Prompt ID': prompt_id,
                    'Metric': metric,
                    'p-value': p_val,
                    'Test Used': test_used
                })

        # Apply multiple testing correction (Benjamini-Hochberg) per metric
        corrected_results = multipletests(p_values, method='fdr_bh')  # False discovery rate correction
        significant_flags = corrected_results[0]  # Boolean array indicating significance
        corrected_pvals = corrected_results[1]  # Corrected p-values

        # Update results with corrected p-values and significance flags
        for i, result in enumerate(results):
            result['Corrected p-value'] = corrected_pvals[i]
            result['Significant'] = significant_flags[i]
            all_results.append(result)  # Append the result to the master list for all metrics

        # Store significant prompts for this metric
        significant_prompts = [result['Prompt ID'] for result in results if result['Significant']]
        all_significant_prompts[metric] = significant_prompts

        print(f"Completed significance testing for {metric}")

    # Save all results to a single CSV file
    p_values_df = pd.DataFrame(all_results)
    p_values_df.to_csv('/', index=False)
    print("All corrected p-values saved to '")

    return all_significant_prompts


# Plotting function to visualize only the significant prompts per metric
def plot_significant_boxplots(data, metrics, fig_title, n_rows, n_cols, start_label_index, all_significant_prompts):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), dpi=450)
    axes = axes.flatten()

    subplot_labels = [chr(i) for i in range(65 + start_label_index, 65 + start_label_index + len(metrics))]

    for i, metric_name in enumerate(metrics):
        # Filter data for significant prompts only
        significant_prompts = all_significant_prompts.get(metric_name, [])
        sns.boxplot(x='id', y='value', hue='condition',  # 'id' used for x-axis
                    data=data[(data['metric'] == metric_name) & (data['id'].isin(significant_prompts))],
                    ax=axes[i], showfliers=False)

        axes[i].set_title(f"{subplot_labels[i]}. Novel vs Base: {metric_name.upper()} (Significant Prompts)",
                          fontsize=16)
        axes[i].set_xticks(range(len(axes[i].get_xticklabels())))
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, fontsize=8)  # Rotate for readability
        axes[i].set_xlabel('Prompt ID', fontsize=12)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].legend(title='Condition', loc='best', fontsize=10)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        # Add red dotted lines between prompts
        for j in range(1, len(significant_prompts)):
            axes[i].axvline(x=j - 0.5, color='red', linestyle='--', linewidth=1)


    plt.suptitle(fig_title, fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"/Users/venus/PycharmProjects/enzymes/statistics/Graphs/{fig_title}.png", format='png', dpi=450,
                bbox_inches='tight')
    print(f"Plot saved to /Users/venus/PycharmProjects/enzymes/statistics/Graphs/{fig_title}.png")


# Load combined data, perform statistical tests, calculate statistics, and plot
group_files = {
    'SSP': {
        'novel': '',
        'base': ''
    },
    'CEP': {
        'novel': '',
        'base': ''
    },
    'CT': {
        'novel': '',
        'base': ''
    },
    'EXP': {
        'novel': '',
        'base': ''
    },
    'POP': {
        'novel': '',
                'base': ''
    },
    'PSP': {
        'novel': '',
        'base': ''
    }
}

combined_data = combine_data(group_files)

# Melt data for easier plotting
melted_data = combined_data.melt(id_vars=['id', 'prompt', 'condition', 'group'],
                                 value_vars=['bleu_score', 'rouge_1', 'rouge_2', 'rouge_L', 'context_similarity'],
                                 var_name='metric', value_name='value')

# Perform statistical tests and get significant prompts for each metric
all_significant_prompts = perform_statistical_tests(melted_data)

# Calculate and save statistics to CSV
calculate_statistics(melted_data, metrics=['bleu_score', 'rouge_1', 'rouge_2', 'rouge_L', 'context_similarity'])

# Plot boxplots for significant prompts only
plot_significant_boxplots(melted_data, metrics=['bleu_score', 'rouge_1', 'rouge_2'],
                          fig_title="Comparison of Significant Individual Prompts (Part 1)",
                          n_rows=1, n_cols=3, start_label_index=0, all_significant_prompts=all_significant_prompts)

plot_significant_boxplots(melted_data, metrics=['rouge_L', 'context_similarity'],
                          fig_title="Comparison of Significant Individual Prompts (Part 2)",
                          n_rows=1, n_cols=2, start_label_index=3, all_significant_prompts=all_significant_prompts)

