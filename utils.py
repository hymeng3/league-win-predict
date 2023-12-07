import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



def read_input(*input_csvs):

	input_dataframe = pd.concat([pd.read_csv(file) for file in input_csvs])
	
	# Drop meaningless columns ('Unnamed: 0', 'matchID')
	# redundant columns ('redFirstBlood', 'redWin')
	# and columns with info we wont have access to for a real game ('fullTimeMS', 'timePercent')
	input_dataframe.drop(columns=['Unnamed: 0', 'matchID', 'redFirstBlood', 'redWin', 'fullTimeMS', 'timePercent'], inplace=True)

	# Convert bools to int
	input_dataframe['blueFirstBlood'] = input_dataframe['blueFirstBlood'].astype(int)
	input_dataframe['blueWin'] = input_dataframe['blueWin'].astype(int)

	return input_dataframe



def feature_transform(dataset):

	transformed_dataset = dataset.copy()
	
	# Create feature for tracking dragon souls
	# Has dragon soul if number of non elder dragon kills = 4
	blueNonElderDragonKills = dataset.blueDragonKill - dataset.blueDragonElderKill
	redNonElderDragonKills = dataset.redDragonKill - dataset.redDragonElderKill
	
	transformed_dataset.insert(len(transformed_dataset.columns), 'blueHasDragonSoul', blueNonElderDragonKills.map(lambda x: 1 if x == 4 else 0))
	transformed_dataset.insert(len(transformed_dataset.columns), 'redHasDragonSoul', redNonElderDragonKills.map(lambda x: 1 if x == 4 else 0))
	
	# Combine blue and red dragon soul, 1 if blue, -1 if red, 0 if neither
	for idx, row in dataset.iterrows():
		if row['redHasDragonSoul'] == 1:
			row['blueHasDragonSoul'] = -1

	# Change minion kills per side to percent difference compared to total kills
	min_minions = 1 # Add 1 min minion to totals to avoid divide by 0
	minionKillDiff = dataset.blueMinionsKilled - dataset.redMinionsKilled
	totalMinionKills = dataset.blueMinionsKilled + dataset.redMinionsKilled + min_minions
	jungleMinionKillDiff = dataset.blueJungleMinionsKilled - dataset.redJungleMinionsKilled
	totalJungleMinionKills = dataset.blueJungleMinionsKilled + dataset.redJungleMinionsKilled + min_minions

	transformed_dataset.insert(len(transformed_dataset.columns), 'percentMinionDiff', minionKillDiff / totalMinionKills)
	transformed_dataset.insert(len(transformed_dataset.columns), 'percentJungleMinionDiff', jungleMinionKillDiff / totalJungleMinionKills)

	# Rescale gold and player levels the same way
	goldDiff = dataset.blueTotalGold - dataset.redTotalGold
	totalGold = dataset.blueTotalGold + dataset.redTotalGold

	avgLevelDiff  = dataset.blueAvgPlayerLevel - dataset.redAvgPlayerLevel
	percentAvgLevelDiff = avgLevelDiff / (dataset.blueAvgPlayerLevel + dataset.redAvgPlayerLevel)

	transformed_dataset.insert(len(transformed_dataset.columns), 'percentAvgLevelDiff', percentAvgLevelDiff)
	transformed_dataset.insert(len(transformed_dataset.columns), 'percentTotalGoldDiff', goldDiff / totalGold)
	
	# Combine neutral objectives and tower kills to percent differences
	heraldKillDiff = dataset.blueRiftHeraldKill - dataset.redRiftHeraldKill
	baronKillDiff = dataset.blueBaronKill - dataset.redBaronKill
	elderKillDiff = dataset.blueDragonElderKill - dataset.redDragonElderKill

	eps = 0.0001 # Small eps to avoid dividing by 0

	avgDragonKillDiff  = dataset.blueDragonKill - dataset.redDragonKill
	percentDragonDiff = avgDragonKillDiff / (dataset.blueDragonKill + dataset.redDragonKill + eps)

	avgTowerKillDiff  = dataset.blueTowerKill - dataset.redTowerKill
	percentTowerDiff = avgTowerKillDiff / (dataset.blueTowerKill + dataset.redTowerKill + eps)

	transformed_dataset.insert(len(transformed_dataset.columns), 'heraldKillDiff', heraldKillDiff)
	transformed_dataset.insert(len(transformed_dataset.columns), 'baronKillDiff', baronKillDiff)
	transformed_dataset.insert(len(transformed_dataset.columns), 'elderKillDiff', elderKillDiff)
	transformed_dataset.insert(len(transformed_dataset.columns), 'percentDragonDiff', percentDragonDiff)
	transformed_dataset.insert(len(transformed_dataset.columns), 'percentTowerDiff', percentTowerDiff)

	# Remove features that have been combined
	combined_features = ['redHasDragonSoul'
						 'blueMinionsKilled',
					  	 'blueJungleMinionsKilled',
                		 'redMinionsKilled',
    					 'redJungleMinionsKilled',
    					 'blueTotalGold',
                		 'redTotalGold',
						 'blueAvgPlayerLevel',
                		 'redAvgPlayerLevel',
                		 'redHasDragonSoul',
                		 'blueRiftHeraldKill',
                		 'blueBaronKill',
                		 'blueDragonElderKill',
                		 'redRiftHeraldKill',
                		 'redBaronKill',
                		 'redDragonElderKill',
						 'blueDragonKill',
                		 'redDragonKill',
                		 'blueTowerKill',
                		 'redTowerKill',
               			]

	transformed_dataset.drop(columns=combined_features, inplace=True)

	# Remove unimportant features
	unimportant_features = ['blueDragonHextechKill',
                		 	'blueDragonChemtechKill',
                		 	'blueDragonFireKill',
                			'blueDragonWaterKill',
                			'blueDragonEarthKill',
                			'blueDragonAirKill',
                			'redDragonHextechKill',
                			'redDragonChemtechKill',
                			'redDragonFireKill',
                			'redDragonWaterKill',
                			'redDragonEarthKill',
                			'redDragonAirKill',
							'blueInhibitorKill',
                        	'redInhibitorKill'
                       		]
	
	transformed_dataset.drop(columns=unimportant_features, inplace=True)

	return transformed_dataset


def scale_dataset(input_set, training_set):
	# Rescale blue/red kills to have mean = 0 and stdev = 1
	# Rescales input based on training set so that the scaling is consistent
	scaler = StandardScaler()
	input_set[['blueChampionKill', 'redChampionKill']] = scaler.fit_transform(training_set[['blueChampionKill', 'redChampionKill']])
	


def process_input(*input_csvs, training_frac=0.8, random_state=RANDOM_SEED):

	# Read data and apply transformations
	input_dataframe = read_input(input_csvs)
	dataset = feature_transform(input_dataframe)

	# Split into training and validation sets
	valid_frac = 1 - training_frac

	train_set, valid_set = train_test_split(dataset, test_size=valid_frac, random_state=RANDOM_SEED)

	train_labels = train_set.pop('blueWin')
	valid_labels = valid_set.pop('blueWin')

	# Scale kills to improve LR and SGD performance
	scaler = StandardScaler()

	for set in [train_set, valid_set]:
		set[['blueChampionKill', 'redChampionKill']] = scaler.fit_transform(train_set[['blueChampionKill', 'redChampionKill']])

	return train_set, train_labels, valid_set, valid_labels





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

