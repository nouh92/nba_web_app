import streamlit as st


def main():
  st.title("Introduction")
  st.markdown(
    """
  **Objectifs du projet**

  Notre projet concerne l'analyse et la prédiction des tirs de 20 des meilleurs joueurs NBA du 21ème siècle. Les technologies actuelles permettent de suivre en temps réel les déplacements de tous les joueurs sur le terrain. De nombreuses données sont disponibles sur Internet.
  Nous allons nous baser sur ces données, avec un double enjeu :

    - Analyse et visualisation : comparaison des 20 joueurs, de la fréquence et de l'efficacité au tir par situation de jeu et par localisation sur le terrain.
    - Modélisation et prédiction : pour chacun des 20 joueurs, estimer la probabilité qu'un tir rentre dans le panier, en fonction de différentes métriques.

  **Expertise**

  Au démarrage du projet, nous n'avons pas d'expertise particulière sur le sujet. Nous connaissons quelques joueurs et équipes, mais peu de vocabulaire et métriques. Les nombreuses informations disponibles sur Internet nous ont permis de mieux appréhender le sujet. On peut citer notamment les sites NBA.com et Basketball Reference.

  **Jeux de données en CSV**

  :blue-background[**NBA shot locations 1997-2020**]

    URL : https://www.kaggle.com/datasets/jonathangmwl/nba-shot-locations/data
    Description : jeu de données sur les tirs au cours des matchs pour toutes les saisons NBA de 1997 à 2020
    Taille du dataset : 4,7+ millions de lignes, 22 colonnes

 :blue-background[**NBA play-by-play data by season 2000-2020**]

    URL : https://sports-statistics.com/sports-data/nba-basketball-datasets-csv-files
    Description : un fichier par saison NBA ; chacun contient les actions de tous les matchs d'une saison.

  :blue-background[**NBA games data 2003-2022**]

    URL : https://www.kaggle.com/datasets/nathanlauga/nba-games
    Description :
      - Statistiques sur les matchs de 2003 à 2022
      - Statistiques par match et par joueur sur la même période de temps
      - Classement de chaque équipe, jour par jour, sur la même période de temps ; nombre de matchs gagnés et perdus par équipe au cours d'une saison.

  :blue-background[**NBA players data 1950-2017**]

    URL : https://www.kaggle.com/datasets/nathanlauga/nba-games?select=players.csv
    Description : statistiques par joueur et par année, sur la période 1950-2017.

  **Sources**

  En accès libre sur deux sources :

    - Kaggle, plateforme web interactive de science des données
    - Sports Statistics qui met à disposition des données relatives à plusieurs sports
    - nba-api : client API pour www.nba.com, un package a pour objectif de rendre les API de NBA.com facilement accessibles

  **Nota bene**

  :blue-background[**L'exploration des données a fait apparaître des données manquantes. Nous avons complété les datasets en récupérant les données sur l'API nba-api.**]

  """
  )
  