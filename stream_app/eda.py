#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
                      ### PAGE Exploratory Data Analysis ###
#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from src.utils import *
from src.classification import *
from src.merge_and_encode import *


def main():
  st.title('Exploratory Data Analysis (EDA)')
  st.write(
    """
    Deux datasets font l'objet d'analyses exploratoires et de visualisations :
    - shots : NBA Shot Locations 1997 - 2020,
    - actions : la fusdion de tous les fichiers des actions pour chaque saison, soit 16 fichiers csv, et des actions manquantes récupérées avec l'API nba-api.
    """
  )


  shots = pd.read_csv('../_DATAS/NBA Shot Locations 1997 - 2020.csv')
  actions = pd.read_csv('../_PRE_PRO_DATAS/actions_full.csv')

  with st.expander("Voir un extrait du jeu de données NBA Shot Locations 1997 - 2020"):
    st.dataframe(
      shots.head(10),
      column_config = {
        'Game ID' : st.column_config.NumberColumn(format='%d'),
        'Team ID' : st.column_config.NumberColumn(format='%d'),
        'Game Date' : st.column_config.NumberColumn(format='%d')
      })
    
  with st.expander("Voir un extrait du jeu de données Actions (cet extrait contient les transformations décrites dans le preprocessing)"):
    st.dataframe(
      actions.head(10),
      column_config = {
        'GAME_ID' : st.column_config.NumberColumn(format='%d'),
        'PLAYER_ID' : st.column_config.NumberColumn(format='%d'),
        'TEAM_ID' : st.column_config.NumberColumn(format='%d'),
        'OPP_TEAM_ID' : st.column_config.NumberColumn(format='%d')
      }
      )



  st.header("I. EDA sur l'ensemble des 20 joueurs")


  #-----------------------------------------------------------------------------------------------------#
  # 0. Initialisation des datas pour la visualisation des zones et des actions
  #-----------------------------------------------------------------------------------------------------#
  top20, df, zones_data, players_df = init_zones_analytics()
  st.subheader("Les zones de Tir")


  #-----------------------------------------------------------------------------------------------------#
  # 1. Viz globales
  #-----------------------------------------------------------------------------------------------------#

  # Viz des zones du terrains
  st.write(
    """
    Les datasets découpent le terrain de basket en zones de tirs selon trois prismes différents :
    - Shot Zone Basic
    - Shot Zone Area
    - Shot Zone Range
  """)
 # display_shot_zones(df)

  col1, col2, col3 = st.columns(3)
  with col1:
    fig = plotly_display_shot_zones(df=df, zone_var='Shot Zone Basic')
    st.plotly_chart(fig)
  with col2:
    fig = plotly_display_shot_zones(df=df, zone_var='Shot Zone Area')
    st.plotly_chart(fig)
  with col3:
    fig = plotly_display_shot_zones(df=df, zone_var='Shot Zone Range')
    st.plotly_chart(fig)

  # Viz de la répartition des tirs par zone
  col1, col2, col3 = st.columns(3)
  with col1:
    fig = plotly_pie_plots(zones_data=zones_data, zone_var='Shot Zone Basic')
    st.plotly_chart(fig)
  with col2:
    fig = plotly_pie_plots(zones_data=zones_data, zone_var='Shot Zone Area')
    st.plotly_chart(fig)
  with col3:
    fig = plotly_pie_plots(zones_data=zones_data, zone_var='Shot Zone Range') 
    st.plotly_chart(fig)

  st.write(
    """
    Il y a 3 zones principales de tir :
    - Restricted Area (zone sous le panier),
    - Mid-Range (entre la raquette et la ligne des 3 points)
    - Above the Break 3 (zone des 3 points).

    Plus d'un tir sur deux est tenté depuis la zone centrale. Près de 40% des tirs sont tentés à moins de 8 pieds, soit 2,5 mètres. 70% des tirs sont tentés dans la zone des 2 points (distance comprise entre 0 et 24 pieds).

    Les boxplots ci-dessous confirment la criticité de la zone centrale dans la réussite des tirs (X Location).
  """)


  missed = zones_data[zones_data['Shot Made Flag'] == 0]
  scored = zones_data[zones_data['Shot Made Flag'] == 1]
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    fig = plotly_boxplot_zones(shot_result="manqués", x = missed['X Location'])
    st.plotly_chart(fig)
  with col2:
    fig = plotly_boxplot_zones(shot_result="réussis", x = scored['X Location'])
    st.plotly_chart(fig)
  with col3:
    fig = plotly_boxplot_zones(shot_result="manqués", y = missed['Y Location'])
    st.plotly_chart(fig)
  with col4:
    fig = plotly_boxplot_zones(shot_result="réussis", y = scored['Y Location'])
    st.plotly_chart(fig)

  st.markdown(
    """
    En analysant plus précisément les localisations des tirs, il se confirme que :blue-background[**la zone où les tirs sont le plus souvent marqués est à une distances faible du panier et centrale**], comme le montre ces visualisations (échantillons de 10000 tirs pour chaque graphique). 
  """)


  fig = plotly_scatter_global_shotZones(zones_data=zones_data)
  st.plotly_chart(fig)


   # Viz de la distribution des variables continues

  st.subheader("Distribution des variables continues")

  st.markdown(
    """
    L'analyse de la distribution des variables continues nous permet deux observations :
      a. la confirmation que la zone de tir est critique dans le résultat
      b. le carctère particulier de la fin de match qu'il conviendra d'analyser plus en détail.

    La variable X Location est celle qui s'approche le plus d'une loi normale.

    La distribution de Y Location est plus complexe, avec un pic fort dans les valeurs faibles de Y Location, puis une première chute pour se stabiliser à un niveau faible, avant une seconde chute vers 0.

    Concernant la fin de match, les variables "Minutes Remaining" et "Seconds Remaining" sont uniformément réparties dans nos données, mais il y a un pic de tirs manqués lorsque le temps de jeu restant est faible.
    
    :blue-background[**Les joueurs augmentent les tentatives de tirs avant la fin du temps de jeu, dégradant ainsi la qualité de leur tir**].
    """
  )

  var_to_plot = ['Y Location', 'X Location', 'Minutes Remaining', 'Seconds Remaining'] 
  distribution_plots(df=df, var_to_plot=var_to_plot, hue='Shot Made Flag',  color1='yellowgreen', color2='lightsteelblue')

  st.write("""
   Les tables de contingence et un test statistique khi-deux permettent de mesurer l'influence de la zone sur le résultat du tir.          
""")
  
  shots_zones = zones_data[['Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Shot Made Flag']]
  var_list = ['Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range']
  target = 'Shot Made Flag'
  with st.expander("Voir les tables de contingence entre les zones et la variable cible"):
    display_contingency(df=shots_zones, columns_list=var_list, target='Shot Made Flag')


  chi2_testing(df=shots_zones, var_list=var_list, target=target)

  st.markdown("""
     Les trois types de zones (Shot Zone Basic, Shot Zone Area, Shot Zone Range) ont une relation avec le résultat du tir car les p-values sont nulles. Ces relations sont d'intensité moyenne comme le montrent les coefficients de Cramer. :blue-background[**C'est la variable "Shot Zone Basic" qui montre la corrélation la plus forte**].
  """)
  zones_cramer_per_player = df_chi2_building(players_dict=players_df, var_list=var_list, target=target)
  zones_cramer_per_player

  st.subheader("Taux de réussite des tirs")

  st.markdown(
    """
    Le graphique suivant nous confirme que le taux de réussite au tir :
    - est un peu moins bon dans le dernier quart-temps ; 
    - est nettement moins bon à la dernière minute de chaque quart-temps, en baisse de 6 points.
    
    :blue-background[**Les tirs tentés à la dernière minute sont probablement plus difficiles, mais il peut aussi y avoir un facteur humain, comme le stress ou l'empressement, qui influe sur le taux de réussite**].
    """
  )

  actions = pd.read_csv('../_PRE_PRO_DATAS/actions_full.csv', index_col=0)
  time_left_viz(actions)


  # Taux de réussite des tirs 2pte et 3 pts

  st.write(
    """
    Les tirs à 3 points sont évidemment moins souvent réussis, mais leur taux de réussite avoisinent cependant les 40% pour nos 20 joueurs, qui figurent parmi les plus performants en NBA.
    """
  )
  shotPtsType = pd.crosstab(zones_data['Shot Type'], zones_data['Shot Made Flag'], normalize=0)
  plot_shotPtsType(shotPtsType=shotPtsType)
  
  # Viz des tirs marqués en fonction de l'action précédente
  st.markdown(
    """
    Globalement, le taux de réussite varie de façon significative en fonction de l'action précédente :
    - Dans la plupart des cas, le taux de réussite est d'environ 45%
    - Si la balle a été récupérée avec un rebond défensif, le taux de réussite est de 48%
    - Si la balle a été récupérée avec un rebond offensif, le taux de réussite est de 50%
    - Si l'adversaire perd la balle (hors rebond défensif), le taux de réussite est de 55%

    Pour certains joueurs, cette différence est encore plus marquée. Par exemple, le taux de réussite de LeBron James est de 63% suite à une perte de balle par un joueur de l'équipe adverse. Pour un joueur comme Rudy Gobert, le taux de réussite est globalement très élevé quelle que soit l'action précédente. Cela s'explique par son profil différent : il fait beaucoup plus de dunks et beaucoup moins de tirs loin du panier.

    L'action précédente semble avoir une influence sur le taux de réussite, mais il est aussi :blue-background[**indispensable d'effectuer des analyses pour chaque joueur car les profils semblent très différents**].
    """
  )
  previous_event_viz(actions)
 

  #-----------------------------------------------------------------------------------------------------#
  # 2. Viz par joueur
  #-----------------------------------------------------------------------------------------------------#

  st.header("II. EDA par joueur")

  st.write("""
    Les visualisations suivantes pour chaque joueur permettent de distinguer des profils différents.
           
    Certains joueurs tentent de nombreux tirs, aussi bien à 2 points qu'à 3 points quand d'autres se spécialisent dans les tirs à 2 points. Les profils sont variés.
    
    En regardant l'évolution de chaque joueur au fil des années, certains se spécialisent : Stephen Curry en début de carrière tire à plus de 60% dans la zone des 2 points, en 2020 près de 80% de ses tirs sont à 3 points.
           
    Autre exemple caractéristique : Rudy Gobert, qui tire uniquement à 2 points depuis le début de sa carrière.        
  """)
  nb_shots_per_player()
  
  shots_split_per_player(players_df)

  st.markdown(
    """
      La visualisation des localisations précises (X et Y) des tirs nous permets d'établir des profils types : 

        - le joueur très majoritairement actif dans la restricted area (sous le panier)
        - le spécialiste des tirs à trois points
        - le joueur polyvalent, actif dans toutes les zones offensives
        - le joueur actif dans la zone des deux points

        Nous constatons que non seulement Rudy Gobert ne tire qu'à 2 points, mais qu'il est quasi exclusivement actif sous la raquette. À l'inverse, Stephen Curry est très présent autour de la zone des 3 points.
              
        :blue-background[**Une zone est presque inactive pour tous les joueurs, celle de l'entrée dans la zone des deux points. Une fois passé la ligne des trois points, il est plus intéressant de s'approcher du panier pour maximiser la probabilité de marquer**].

           
    """)
  

  fig = plotly_playerZones(players_df)
  st.plotly_chart(fig)

  st.write("""
      Certains joueurs enregistrent des niveaux de performances contrastés selon les années.
           
      En faisant un double-clic sur un joueur, le graphique affiche la courbe de celui-ci ; il est possible d'afficher un ou plusieurs joueurs.
""")
  
  success_rate_per_year(players_df)
  

