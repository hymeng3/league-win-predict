import requests
import pandas as pd

# Get match history
def get_match_history(api_key, game_name, tagline, start=0, count=20):

	# Get puuid from name + tagline
	account_url = "https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/" + game_name + "/" + tagline
	account_url = account_url + '?api_key=' + api_key
	account = requests.get(account_url)
	account_info = account.json()
	puuid = account_info['puuid']

	# Get match history
	matches_url = "https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/"+puuid+"/ids?start="+start+"&count="+count
	matches_url = matches_url + '&api_key=' + api_key
	matches = requests.get(matches_url)
	return matches.json()

# Get data from single match
def get_match_info(api_key, match_id):

	# Get match info from api
	match_url = "https://americas.api.riotgames.com/lol/match/v5/matches/"+match_id
	match_url = match_url + '?api_key=' + api_key
	resp = requests.get(match_url)
	match_info = resp.json()

	# Extract data into dictionary
	data = {}

	data['blueWin'] = match_info['info']['teams'][0]['win']
	data['blueChampionKill'] = match_info['info']['teams'][0]['objectives']['champion']['kills']
	data['blueFirstBlood'] = match_info['info']['teams'][0]['objectives']['champion']['first']
	data['blueDragonKill'] = match_info['info']['teams'][0]['objectives']['dragon']['kills']
	data['blueRiftHeraldKill'] = match_info['info']['teams'][0]['objectives']['riftHerald']['kills']
	data['blueBaronKill'] = match_info['info']['teams'][0]['objectives']['baron']['kills']
	data['blueTowerKill'] = match_info['info']['teams'][0]['objectives']['tower']['kills']
	data['blueInhibitorKill'] = match_info['info']['teams'][0]['objectives']['inhibitor']['kills']
	data['blueDragonElderKill'] = match_info['info']['participants'][0]['challenges']['teamElderDragonKills']

	data['redChampionKill'] = match_info['info']['teams'][1]['objectives']['champion']['kills']
	data['redFirstBlood'] = match_info['info']['teams'][1]['objectives']['champion']['first']
	data['redDragonKill'] = match_info['info']['teams'][1]['objectives']['dragon']['kills']
	data['redRiftHeraldKill'] = match_info['info']['teams'][1]['objectives']['riftHerald']['kills']
	data['redBaronKill'] = match_info['info']['teams'][1]['objectives']['baron']['kills']
	data['redTowerKill'] = match_info['info']['teams'][1]['objectives']['tower']['kills']
	data['redInhibitorKill'] = match_info['info']['teams'][1]['objectives']['inhibitor']['kills']
	data['redDragonElderKill'] = match_info['info']['participants'][0]['challenges']['teamElderDragonKills']


	blueTotalGold = 0
	blueMinionsKilled = 0
	blueJungleMinionsKilled = 0
	blueAvgPlayerLevel = 0

	for i in range(5):
		blueTotalGold += match_info['info']['participants'][i]['goldEarned']
		blueMinionsKilled += match_info['info']['participants'][i]['totalMinionsKilled']
		blueJungleMinionsKilled += match_info['info']['participants'][i]['totalAllyJungleMinionsKilled']
		blueJungleMinionsKilled += match_info['info']['participants'][i]['totalEnemyJungleMinionsKilled']
		blueAvgPlayerLevel += match_info['info']['participants'][i]['champLevel'] / 5

	redTotalGold = 0
	redMinionsKilled = 0
	redJungleMinionsKilled = 0
	redAvgPlayerLevel = 0

	for i in range(5, 10):
		redTotalGold += match_info['info']['participants'][i]['goldEarned']
		redMinionsKilled += match_info['info']['participants'][i]['totalMinionsKilled']
		redJungleMinionsKilled += match_info['info']['participants'][i]['totalAllyJungleMinionsKilled']
		redJungleMinionsKilled += match_info['info']['participants'][i]['totalEnemyJungleMinionsKilled']
		redAvgPlayerLevel += match_info['info']['participants'][i]['champLevel'] / 5

	data['blueTotalGold'] = blueTotalGold
	data['blueMinionsKilled'] = blueMinionsKilled
	data['blueJungleMinionsKilled'] = blueJungleMinionsKilled
	data['blueAvgPlayerLevel'] = blueAvgPlayerLevel

	data['redTotalGold'] = redTotalGold
	data['redMinionsKilled'] = redMinionsKilled
	data['redJungleMinionsKilled'] = redJungleMinionsKilled
	data['redAvgPlayerLevel'] = redAvgPlayerLevel

	df = pd.DataFrame.from_dict(data)

	return df

