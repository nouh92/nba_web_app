import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

def rename_and_filter_vars(df):
    
    var_names = {'Shot Distance' : 'shotDistance',
                 'X Location' : 'Xloc',
                 'Y Location' : 'Yloc',
                 'Shot Zone Basic' : 'shotZoneBasic',
                 'Shot Zone Area' : 'shotZoneArea',
                 'Shot Zone Range' : 'shotZoneRange',
                 'Action Type' : 'shotType',
                 'Shot Type' : 'shotPoints',
                 'Player Name' : 'player',
                 'year' : 'year',
                 'month' : 'month',
                 'HOME' : 'home',
                 'PERIOD' : 'period',
                 'MINS_LEFT' : 'minsLeft',
                 'PREVIOUS_EVENT' : 'previousEvent',
                 'W_PCT_x' : 'winPercentage',
                 'W_PCT_y' : 'oppWinPercentage',
                 'Shot Made Flag' : 'target'}
    
    return df.rename(columns=var_names).filter(items=list(var_names.values()))

def filter_before_merge(actions, shots):
    
    # Suppression des actions absentes du dataset des tirs
    actions_fg = actions[(actions.EVENT == 'Field goal made') | (actions.EVENT == 'Field goal attempt')]
    unmatch = actions_fg[~actions_fg.index.isin(shots.index)]
    actions_filtered = actions[~actions.index.isin(unmatch.index)]
    # print("Suppression de", len(unmatch), "actions absentes du dataset des tirs")
    
    # Suppression des tirs absents du dataset des actions
    unmatch = shots[~shots.index.isin(actions_fg.index)]
    shots_filtered = pd.DataFrame()
    if len(unmatch) < 1000: # on supprime les tirs seulement s'il y en a peu à supprimer = si le dataset des actions est complet
        shots_filtered = shots[~shots.index.isin(unmatch.index)]
        # print("Suppression de", len(unmatch), "tirs absents du dataset des actions")
    
    return actions_filtered, shots_filtered

def get_last_period(df):
    
    # Calcul de la variable lastPeriod
    # Valorisée à 0 pour les 3 premières périodes et à 1 pour la dernière période et les prolongations
    last_period = [0 for i in range(1,4)]
    last_period.extend([1 for i in range(4,10)])
    last_period = pd.Series(last_period, index=range(1,10))  
    df['lastPeriod'] = df.period.map(last_period)
    df = df.drop(columns='period')
    
    return df

def get_last_min(df):
    
    # Calcul de la variable lastMin (1 = dernière minute d'une période de jeu)
    last_min = [1]
    last_min.extend([0 for i in range(1,13)])
    last_min = pd.Series(last_min, index=range(0,13))  
    df['lastMin'] = df.minsLeft.map(last_min)    
    df = df.drop(columns='minsLeft')
    
    return df

def get_player_average(df):
    
    # Taux de réussite moyen par joueur sur l'ensemble des tirs passés
    # Pour le premier tir, on met le taux de réussite moyen pour l'ensemble des joueurs
    df['playerAverage'] = df.groupby(['player'])['target'].transform(lambda x: x.expanding().mean().shift(1))
    df['playerAverage'] = df['playerAverage'].fillna(round(df.target.mean(), 3))
    
    return df

def get_player_shape(df, recentShots):
    
    # Taux de réussite moyen par joueur sur les tirs récents
    df['playerShape'] = df.groupby(['player'])['target'].transform(lambda x: x.rolling(window=recentShots).mean().shift(1))
    df['playerShape'] = df['playerShape'].fillna(df['playerAverage'])
    
    return df

def one_hot_encoding(df, cols_to_encode):
    
    encoder = OneHotEncoder(sparse_output=False, dtype='int')
    df_encoded = pd.DataFrame(data=encoder.fit_transform(df.filter(items=cols_to_encode)),
                              columns=encoder.get_feature_names_out(), index=df.index)
    df = pd.concat([df.drop(columns=cols_to_encode), df_encoded], axis=1)
    
    return df

def get_win_percentages(df, rankings):
    
    # Pourcentage de victoires de l'équipe du joueur à la veille du match
    # Il s'agit du pourcentage depuis le début de la saison (égal à 0 en début de saison)
    df = df.merge(rankings[['TEAM_ID', 'STANDINGSDATE', 'W_PCT']],
                  left_on=['TEAM_ID', 'Shifted Date'],
                  right_on=['TEAM_ID', 'STANDINGSDATE'])
    
    # Pourcentage de victoires de l'équipe adverse à la veille du match
    # Il s'agit du pourcentage depuis le début de la saison (égal à 0 en début de saison)
    df = df.merge(rankings[['TEAM_ID', 'STANDINGSDATE', 'W_PCT']],
                  left_on=['OPP_TEAM_ID', 'Shifted Date'],
                  right_on=['TEAM_ID', 'STANDINGSDATE'])
    
    return df
