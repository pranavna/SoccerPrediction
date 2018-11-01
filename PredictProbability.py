import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def get_match_features(match, matches, x = 10):
    ''' Create match specific features for a given match. '''
    
    
    #Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id
    
    
    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x = 10)
    matches_away_team = get_last_matches(matches, date, away_team, x = 10)
    
    #Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 2)
    
    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)
    
    #Define result data frame
    result = pd.DataFrame()
    
    #Define ID features
    #result.loc[0, 'match_api_id'] = match.match_api_id
    #result.loc[0, 'league_id'] = match.league_id

    #Create match features
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team) 
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    
    #Return match features
    return result.loc[0]

def get_last_matches(matches, date, team, x = 10):
    ''' Get the last x matches of a given team. '''
    
    
    #Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
                           
    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    
    #Return last matches
    return last_matches

def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    ''' Get the last x matches of two given teams. '''
    
    
    #Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]    
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]  
    total_matches = pd.concat([home_matches, away_matches])
    
    #Get last x matches
    try:    
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:total_matches.shape[0],:]
        
        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")
            
    #Return data
    return last_matches

def get_goals(matches, team):
    ''' Get the goals of a specfic team from a set of matches. '''
    
    
    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals
    
    #Return total goals
    return total_goals

def get_goals_conceided(matches, team):
    ''' Get the goals conceided of a specfic team from a set of matches. '''

    
    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    #Return total goals
    return total_goals

def get_wins(matches, team):
    ''' Get the number of wins of a specfic team from a set of matches. '''
    
    
    #Find home and away wins
    home_wins = int(matches.home_team_goal[(matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[(matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    #Return total wins
    return total_wins      




def main():
  
	args = sys.argv[1:]
	
	warnings.simplefilter("ignore")

	df_new = pd.read_csv('New Matches API.csv', header = 0)
	df_team = pd.read_csv('team.csv',header = 0)
	df_player = pd.read_csv('player.csv',header = 0)

	df_predict = pd.DataFrame(columns = ['home_team_goals_difference','away_team_goals_difference','games_won_home_team','games_won_away_team','games_against_won','games_against_lost','League_21518.0','League_24558.0','home_player_1_overall_rating','home_player_2_overall_rating','home_player_3_overall_rating','home_player_4_overall_rating','home_player_5_overall_rating','home_player_6_overall_rating','home_player_7_overall_rating','home_player_8_overall_rating','home_player_9_overall_rating','home_player_10_overall_rating','home_player_11_overall_rating','away_player_1_overall_rating','away_player_2_overall_rating','away_player_3_overall_rating','away_player_4_overall_rating','away_player_5_overall_rating','away_player_6_overall_rating','away_player_7_overall_rating','away_player_8_overall_rating','away_player_9_overall_rating','away_player_10_overall_rating','away_player_11_overall_rating','B365_Win','B365_Draw','B365_Defeat','BW_Win','BW_Draw','BW_Defeat'])

	ht = df_team.team_api_id[df_team['team_long_name']== args[28]].values[0]
	at = df_team.team_api_id[df_team['team_long_name']== args[29]].values[0]

	h1 = args[0] #'De Gea'
	h2 = args[1] #'L. Shaw'
	h3 = args[2] #'C. Smalling'
	h4 = args[3] #'E. Bailly'
	h5 = args[4] #'A. Valencia'
	h6 = args[5] #'M. Rashford'
	h7 = args[6] #'Juan Mata'
	h8 = args[7] #'P. Pogba'
	h9 = args[8] #'A. Martial'
	h10 = args[9] #'Ander Herrera'
	h11 = args[10] #'R. Lukaku'

	a1 = args[11] #'L. Karius'
	a2 = args[12] #'N. Clyne'
	a3 = args[13] #'J. Matip'
	a4 = args[14] #'V. van Dijk'
	a5 = args[15] #'J. Milner'
	a6 = args[16] #'E. Can'
	a7 = args[17] #'A. Lallana'
	a8 = args[18] #'J. Henderson'
	a9 = args[19] #'G. Wijnaldum'
	a10 = args[20] #'M. Salah'
	a11 = args[21] #'Roberto Firmino'

	df_predict.loc[0,'B365_Win'] = args[22] #0.42
	df_predict.loc[0,'B365_Draw'] = args[23] #0.23
	df_predict.loc[0,'B365_Defeat'] = args[24] #0.35

	df_predict.loc[0,'BW_Win'] = args[25] #0.44
	df_predict.loc[0,'BW_Draw'] = args[26] #0.26
	df_predict.loc[0,'BW_Defeat'] = args[27] #0.30

	df_predict.loc[0,'League_21518.0'] = 1
	df_predict.loc[0,'League_24558.0'] = 0

	
	
	
	home_player_1_overall_rating = df_player.Overall[df_player.player_name==h1]
	home_player_2_overall_rating = df_player.Overall[df_player.player_name==h2]
	home_player_3_overall_rating = df_player.Overall[df_player.player_name==h3]
	home_player_4_overall_rating = df_player.Overall[df_player.player_name==h4]
	home_player_5_overall_rating = df_player.Overall[df_player.player_name==h5]
	home_player_6_overall_rating = df_player.Overall[df_player.player_name==h6]
	home_player_7_overall_rating = df_player.Overall[df_player.player_name==h7]
	home_player_8_overall_rating = df_player.Overall[df_player.player_name==h8]
	home_player_9_overall_rating = df_player.Overall[df_player.player_name==h9]
	home_player_10_overall_rating = df_player.Overall[df_player.player_name==h10]
	home_player_11_overall_rating = df_player.Overall[df_player.player_name==h11]

	away_player_1_overall_rating = df_player.Overall[df_player.player_name==a1]
	away_player_2_overall_rating = df_player.Overall[df_player.player_name==a2]
	away_player_3_overall_rating = df_player.Overall[df_player.player_name==a3]
	away_player_4_overall_rating = df_player.Overall[df_player.player_name==a4]
	away_player_5_overall_rating = df_player.Overall[df_player.player_name==a5]
	away_player_6_overall_rating = df_player.Overall[df_player.player_name==a6]
	away_player_7_overall_rating = df_player.Overall[df_player.player_name==a7]
	away_player_8_overall_rating = df_player.Overall[df_player.player_name==a8]
	away_player_9_overall_rating = df_player.Overall[df_player.player_name==a9]
	away_player_10_overall_rating = df_player.Overall[df_player.player_name==a10]
	away_player_11_overall_rating = df_player.Overall[df_player.player_name==a11]

	df_predict.loc[0,'home_player_1_overall_rating'] = df_player.Overall[df_player.player_name==h1].values[0]
	df_predict.loc[0,'home_player_2_overall_rating'] = df_player.Overall[df_player.player_name==h2].values[0]
	df_predict.loc[0,'home_player_3_overall_rating'] = df_player.Overall[df_player.player_name==h3].values[0]
	df_predict.loc[0,'home_player_4_overall_rating'] = df_player.Overall[df_player.player_name==h4].values[0]
	df_predict.loc[0,'home_player_5_overall_rating'] = df_player.Overall[df_player.player_name==h5].values[0]
	df_predict.loc[0,'home_player_6_overall_rating'] = df_player.Overall[df_player.player_name==h6].values[0]
	df_predict.loc[0,'home_player_7_overall_rating'] = df_player.Overall[df_player.player_name==h7].values[0]
	df_predict.loc[0,'home_player_8_overall_rating'] = df_player.Overall[df_player.player_name==h8].values[0]
	df_predict.loc[0,'home_player_9_overall_rating'] = df_player.Overall[df_player.player_name==h9].values[0]
	df_predict.loc[0,'home_player_10_overall_rating'] = df_player.Overall[df_player.player_name==h10].values[0]
	df_predict.loc[0,'home_player_11_overall_rating'] = df_player.Overall[df_player.player_name==h11].values[0]

	df_predict.loc[0,'away_player_1_overall_rating'] = df_player.Overall[df_player.player_name==a1].values[0]
	df_predict.loc[0,'away_player_2_overall_rating'] = df_player.Overall[df_player.player_name==a2].values[0]
	df_predict.loc[0,'away_player_3_overall_rating'] = df_player.Overall[df_player.player_name==a3].values[0]
	df_predict.loc[0,'away_player_4_overall_rating'] = df_player.Overall[df_player.player_name==a4].values[0]
	df_predict.loc[0,'away_player_5_overall_rating'] = df_player.Overall[df_player.player_name==a5].values[0]
	df_predict.loc[0,'away_player_6_overall_rating'] = df_player.Overall[df_player.player_name==a6].values[0]
	df_predict.loc[0,'away_player_7_overall_rating'] = df_player.Overall[df_player.player_name==a7].values[0]
	df_predict.loc[0,'away_player_8_overall_rating'] = df_player.Overall[df_player.player_name==a8].values[0]
	df_predict.loc[0,'away_player_9_overall_rating'] = df_player.Overall[df_player.player_name==a9].values[0]
	df_predict.loc[0,'away_player_10_overall_rating'] = df_player.Overall[df_player.player_name==a10].values[0]
	df_predict.loc[0,'away_player_11_overall_rating'] = df_player.Overall[df_player.player_name==a11].values[0]

	df_new.drop('Unnamed: 0',axis=1, inplace = True)
	df_new = df_new.append({'home_team_api_id':ht, 'away_team_api_id':at, 'date':'25-04-2018'}, ignore_index = True)
	df_new.sort_index(inplace=True)
	match = df_new.iloc[-1,:]
	match_stats = get_match_features(match, df_new, x = 10)
	
	df_predict.loc[0,'home_team_goals_difference'] = match_stats.home_team_goals_difference
	df_predict.loc[0,'away_team_goals_difference'] = match_stats.away_team_goals_difference
	df_predict.loc[0,'games_won_home_team'] = match_stats.games_won_home_team
	df_predict.loc[0,'games_won_away_team'] = match_stats.games_won_away_team
	df_predict.loc[0,'games_against_won'] = match_stats.games_against_won
	df_predict.loc[0,'games_against_lost'] = match_stats.games_against_lost
	
	clf = joblib.load('LogisticRegression.pkl') 
	clf2 = joblib.load('NaiveBayes.pkl') 
	clf3 = joblib.load('SVM.pkl') 
	
	print('Logistic Regression Predictions:')
	print(clf.predict_proba(df_predict))
	print(clf.classes_)
	print()
	
	print('Naive Bayes Predictions:')
	print(clf2.predict_proba(df_predict))
	print(clf2.classes_)
	print()
	
	print('SVM Predictions:')
	print(clf3.predict_proba(df_predict))
	print(clf3.classes_)
	print()
	
if __name__ == '__main__':
	main()