import pandas as pd

def get_inputs():
	df1 = pd.read_csv('data/full_data_20.csv')
	df2 = pd.read_csv('data/full_data_40.csv')
	df3 = pd.read_csv('data/full_data_60.csv')
	df4 = pd.read_csv('data/full_data_80.csv')
	df5 = pd.read_csv('data/full_data_100.csv')
	
	game_states = [df1, df2, df3, df4, df5]
	
	for df in game_states:
		df.drop(columns=['Unnamed: 0', 'matchID', 'fullTimeMS', 'timePercent', 'redFirstBlood', 'redWin'], inplace=True)
		df['blueFirstBlood'] = df['blueFirstBlood'].astype(int)
		
	return df1, df2, df3, df4, df5
