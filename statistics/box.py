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
def calculate_boxplot_statistics(data, metrics):
    stats_list = []
    for metric in metrics:
        for group in data['group'].unique():
            for condition in data['condition'].unique():
                metric_data = data[(data['metric'] == metric) &
                                   (data['group'] == group) &
                                   (data['condition'] == condition)]['value'].dropna()
                if not metric_data.empty:
                    stats = {
                        'Metric': metric,
                        'Group': group,
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
    stats_df.to_csv('/',
                    index=False)
    print("Boxplot statistics saved to 'statistics/data_output/group_boxplot_statistics'")
    return stats_df


# Function to run statistical tests for each group and apply multiple testing correction
def perform_statistical_tests(data):
    p_values = []

    for metric in data['metric'].unique():
        for group in data['group'].unique():
            novel_values = data[(data['metric'] == metric) &
                                (data['group'] == group) &
                                (data['condition'] == 'Novel')]['value'].dropna()

            base_values = data[(data['metric'] == metric) &
                               (data['group'] == group) &
                               (data['condition'] == 'Base')]['value'].dropna()

            if len(novel_values) == len(base_values) and len(novel_values) > 0:
                # Check for normality using Shapiro-Wilk test
                stat_novel, p_novel = shapiro(novel_values)
                stat_base, p_base = shapiro(base_values)

                if p_novel > 0.05 and p_base > 0.05:
                    # If both are normally distributed, use paired t-test
                    stat, p_val = ttest_rel(novel_values, base_values)
                    test_used = 't-test'
                else:
                    # If not normally distributed, use Wilcoxon signed-rank test
                    stat, p_val = wilcoxon(novel_values, base_values)
                    test_used = 'Wilcoxon'

                p_values.append({
                    'Metric': metric,
                    'Group': group,
                    'p-value': p_val,
                    'Test Used': test_used
                })

    # Convert p-values to a DataFrame
    p_values_df = pd.DataFrame(p_values)

    # Perform multiple testing correction per metric
    for metric in p_values_df['Metric'].unique():
        metric_pvals = p_values_df[p_values_df['Metric'] == metric]['p-value']
        corrected = multipletests(metric_pvals, method='fdr_bh')  # Benjamini-Hochberg correction
        p_values_df.loc[p_values_df['Metric'] == metric, 'corrected_p-value'] = corrected[1]
        p_values_df.loc[p_values_df['Metric'] == metric, 'significant'] = corrected[0]

    # Save the corrected p-values to a CSV file
    p_values_df.to_csv('/statistics/data_output/p_values_corrected.csv',
                       index=False)
    print("P-values with corrections saved to 'p_values_corrected.csv'")
    return p_values_df


# Plotting function for significant groups per metric
def plot_significant_groups(data, metrics, p_values_df, fig_title, n_rows, n_cols, start_label_index):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), dpi=450)
    axes = axes.flatten()

    subplot_labels = [chr(i) for i in range(65 + start_label_index, 65 + start_label_index + len(metrics))]

    for i, metric_name in enumerate(metrics):
        # Filter significant groups for the current metric
        significant_groups = p_values_df[(p_values_df['Metric'] == metric_name) &
                                         (p_values_df['significant'] == True)]['Group'].unique()

        if len(significant_groups) == 0:
            print(f"No significant groups to plot for {metric_name}.")
            axes[i].axis('off')  # Hide the subplot if no significant groups
            continue

        sns.boxplot(x='group', y='value', hue='condition',
                    data=data[(data['metric'] == metric_name) & (data['group'].isin(significant_groups))],
                    ax=axes[i])
        axes[i].set_title(f"{subplot_labels[i]}. Novel vs Base: {metric_name.upper()} (Significant Groups)",
                          fontsize=16)
        axes[i].set_xticks(range(len(axes[i].get_xticklabels())))
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, fontsize=12)
        axes[i].set_xlabel('Group', fontsize=14)
        axes[i].set_ylabel('Score', fontsize=14)
        axes[i].legend(title='Condition', loc='best', fontsize=12)

    plt.suptitle(fig_title, fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"/{fig_title}.png", format='png', dpi=450,
                bbox_inches='tight')
    # plt.show()


# Load combined data, perform statistical tests, calculate statistics, and plot
group_files = {
    'SSP': {
        'novel': 'path/to/novel_SSP_file.json',
        'base': 'path/to/base_SSP_file.json'
    },
    'CEP': {
        'novel': 'path/to/novel_CEP_file.json',
        'base': 'path/to/base_CEP_file.json'
    },
    'CT': {
        'novel': 'path/to/novel_CT_file.json',
        'base': 'path/to/base_CT_file.json'
    },
    'EXP': {
        'novel': 'path/to/novel_EXP_file.json',
        'base': 'path/to/base_EXP_file.json'
    },
    'POP': {
        'novel': 'path/to/novel_POP_file.json',
        'base': 'path/to/base_POP_file.json'
    },
    'PSP': {
        'novel': 'path/to/novel_PSP_file.json',
        'base': 'path/to/base_PSP_file.json'
    }
}

combined_data = combine_data(group_files)

# Melt data for easier plotting
melted_data = combined_data.melt(id_vars=['id', 'prompt', 'condition', 'group'],
                                 value_vars=['bleu_score', 'rouge_1', 'rouge_2', 'rouge_L', 'context_similarity'],
                                 var_name='metric', value_name='value')

# Perform statistical tests and get p-values with correction
p_values_df = perform_statistical_tests(melted_data)

# Save boxplot statistics for the groups
calculate_boxplot_statistics(melted_data, metrics=['bleu_score', 'rouge_1', 'rouge_2', 'rouge_L', 'context_similarity'])

# Plot boxplots for significant groups only
plot_significant_groups(melted_data, metrics=['bleu_score', 'rouge_1', 'rouge_2'],
                        p_values_df=p_values_df,
                        fig_title="Comparison of Significant Prompt Groups (Part 1)",
                        n_rows=1, n_cols=3, start_label_index=0)

plot_significant_groups(melted_data, metrics=['rouge_L', 'context_similarity'],
                        p_values_df=p_values_df,
                        fig_title="Comparison of Significant Prompt Groups (Part 2)",
                        n_rows=1, n_cols=2, start_label_index=3)

