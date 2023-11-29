import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_inputs():
	df1 = pd.read_csv('data/full_data_20.csv')
	df2 = pd.read_csv('data/full_data_40.csv')
	df3 = pd.read_csv('data/full_data_60.csv')
	df4 = pd.read_csv('data/full_data_80.csv')
	df5 = pd.read_csv('data/full_data_100.csv')
	
	game_states = [df1, df2, df3, df4, df5]
	

    # Drop redundant columns
	for df in game_states:
		df.drop(columns=['Unnamed: 0', 'matchID', 'redFirstBlood', 'redWin', 'fullTimeMS', 'timePercent'], inplace=True)
		df['blueFirstBlood'] = df['blueFirstBlood'].astype(int)
		df['blueWin'] = df['blueWin'].astype(int)
		
	return df1, df2, df3, df4, df5


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

def get_batch(data, labels, batch_size):
    return tf.data.Dataset.from_tensor_slices((data,labels)).batch(batch_size)
