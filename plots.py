import matplotlib.pyplot as plt
import seaborn as sns
import math


"""
Various plotting functions used in EDA and analysis of model performance
"""

def corr_scatter_plot( dataset, x_column, hue_column, alpha=1):
    for column in dataset.columns:
        if (column != x_column.name and column != hue_column.name):
            corr_scatter = sns.scatterplot(data=dataset, x=x_column, y=column, hue=hue_column, alpha=alpha)
            plt.show()

def binary_comparison_plot( dataset, target):
	cont_cols = [f for f in dataset.columns if dataset[f].dtype != '0']
	n_rows = len(cont_cols)

	fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))

	for i, col in enumerate(cont_cols):
			sns.violinplot(x=target, y=col, data=dataset, ax=axs[i], inner="quart", bw_adjust = 1.1)
			axs[i].set_title(f'{col.title()} Distribution by Target', fontsize=14)
			axs[i].set_xlabel(target, fontsize=12)
			axs[i].set_ylabel(col.title(), fontsize=12)

	fig.tight_layout()

	plt.show()

def get_win_percent(n_bins, input_list):

    pred_true = [0 for i in range(n_bins)]
    pred_total = [0 for i in range(n_bins)]
    

    for i in range(len(input_list)):
        idx = min(math.floor(input_list[i][0] * n_bins), n_bins - 1)
        pred_total[idx] += 1
        if input_list[i][1] == 1:
            pred_true[idx] += 1

    bin_content = [pred_true[i] / pred_total[i] for i in range(n_bins)]

    return bin_content

def get_bin_centers(n_bins, input_list):
      
    expected = [0 for i in range(n_bins)]
    pred_total = [0 for i in range(n_bins)]

    for i in range(len(input_list)):
        idx = min(math.floor(input_list[i][0] * n_bins), n_bins - 1)
        expected[idx] += input_list[i][0]
        pred_total[idx] += 1

    expected = [expected[i] / pred_total[i] for i in range(n_bins)]

    return expected

def get_residuals(n_bins, input_list):

    expected = [0 for i in range(n_bins)]
    pred_total = [0 for i in range(n_bins)]
    std = [0 for i in range(n_bins)]
    residuals = [0 for i in range(n_bins)]

    for i in range(len(input_list)):
        idx = min(math.floor(input_list[i][0] * n_bins), n_bins - 1)
        expected[idx] += input_list[i][0]
        pred_total[idx] += 1

    expected = [expected[i] / pred_total[i] for i in range(n_bins)]

    for i in range(len(input_list)):
        idx = min(math.floor(input_list[i][0] * n_bins), n_bins - 1)
        std[idx] += (input_list[i][0] - expected[idx]) ** 2

    std = [math.sqrt(std[i]/pred_total[i]) for i in range(n_bins)]

    observed = get_win_percent(n_bins, input_list)
    residuals = [(observed[i] - expected[i]) / std[i] for i in range(n_bins)]

    return residuals