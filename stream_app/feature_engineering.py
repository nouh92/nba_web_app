#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
                      ### PAGE preprocessing & Feature Engineering ###
#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

import streamlit as st
import pandas as pd

def main() :
  st.title('Preprocessing & Feature Engineering')

  st.write(
    """
    L'EDA réalisée précedemment nous a permis de mieux comprendre nos jeux de données, d'y mettre du sens et de sélectionner les variables les plus importantes pour notre projet de machine learning.
    L'objectif des étapes de preprocessing et de feature engineering est désormais de construire un dataframe utilisable pour les modèles de machine learning.
    """
  )

  st.subheader('I. Preprocessing')

  st.write("Comme nous avons travaillé sur deux dataframes principaux, ainsi que sur un troisième dans un second temps, il nous a fallu préparer ceux-ci à être fusionnés.")

  with st.expander('Voir le détail des étapes de preprocessing'):
    st.markdown(
      """
      **Datasets NBA Shot Locations 1997 - 2020**
      - Filtre sur les 20 joueurs choisis
      - Création d'un index avec l'identifiant du match et le numéro d'événement dans le match, ce qui permettra un merge avec le dataset des actions
      - Suppression des données aberrantes : tirs à 2 points dans une zone à 3 points, ou inversement (34 lignes supprimées)
      - Split de la date du match : variables "year" et "month"
      - Export du résultat dans un fichier CSV prêt à être mergé
      
      **Les datase des actions par saison**
      - Filtre sur les 20 joueurs choisis
      - Création d'un index avec l'identifiant du match et le numéro d'événement dans le match, ce qui permettra un merge avec le dataset destirs
      - Ajoute d'un libellé pour chaque type d'événement, pour plus de lisibilité
      - Pour les lancers-francs, on précise s'ils sont réussis ou manqués (en regardant si le score évolue ou non)
      - Pour les rebonds, on précise s'ils sont offensifs ou défensifs (en regardant l'événement précédent)
      - Ajout  d'une colonne pour indiquer l'événement précédent
      - Définition du joueur, de l'équipe à domicile et de l'équipe adverse
      - Calcule de la marge au score
      - Calcule du nombre de minutes restantes
      - Conservation des tirs dans le jeu et des lancers-francs et suppression des autres actions : rebonds, pertes de balles, ...
      - Conservation des colonnes utiles pour la suite

      Sur un total de 8837 matchs, 1307 matchs sont présents dans le dataset des tirs et absents du dataset des actions. Les données manquantes ont été téléchargées en utilisant l'API nba_api, disponible sur le repository GitHub suivant : https://github.com/swar/nba_api. 

      Le téléchargement s'est fait facilement et rapidement :
      """
    )
    col1, _ = st.columns([4,1])
    col1.image("api_playbyplayv2.png")

  st.subheader('II. Feature engineering')
  st.markdown(
    """
    Les variables suivantes, présentes dans le dataset Shot Locations, sont utilisées pour la modélisation. À noter que ces variables ont été renommées lors du merge : la convention "camel case" est utilisée et quelques noms de variables ont été modifiés.
    """
  )

  with st.expander('Voir la liste des variables de Shot Locations utilisées'):
        st.write(
          """
          - :blue-background[**shotDistance**] : distance par rapport au centre du panier, en dixièmes de pieds (environ 3 cm).

          - :blue-background[**Xloc**] : emplacement par rapport au centre du panier, dans l'axe de la largeur, en dixièmes de pieds. Le terrain fait 50 pieds de large (15,2 mètres). Cette variable varie entre -250 et +250. Quand le joueur fait face au panier et que le panier se site à sa gauche, Xloc est négatif. Dans le cas contraire, Xloc est positif.

          - :blue-background[**Yloc**] : emplacement par rapport au centre du panier, dans l'axe de la longueur, en dixièmes de pieds. Le terrain fait 94 pieds de longueur (28,6 mètres). Une moitié de terrain fait donc 47 pieds de longueur. Dans le dataset, les modalités de cette variable sont comprises entre -82 et +884. Il peut arriver que le tir soit déclenché un peu à l'arrière du centre du panier, ce qui explique les valeurs négatives.

          - :blue-background[**shotZoneBasic**] : découpage du terrain en zones de tir : Restricted Area, Mid-Range, Above the Break 3, In The Paint (Non-RA), Left Corner, Right Corner 3, Backcourt. Voir paragraphe 4.1 : visualisation des données du dataset des tirs.

          - :blue-background[**shotZoneArea**] : découpage du terrain en rectangles : Center(C), Left Side(L), Right Side(R), Right Side Center(RC), Left Side Center(LC), Back Court(BC). Voir paragraphe 4.1.

          - :blue-background[**shotZoneRange**] : distance de tir par zones : Less Than 8 ft, 8-16 ft, 16-24 ft, 24+ ft, Back Court Shot. Voir paragraphe 4.1.

          - :blue-background[**shotType**] : type de tir : Jump Shot, Layup Shot, etc. Le dataset contient 70 types de tirs différents. Les tirs de type "Jump Shot" sont les plus fréquents et représentent 43% des tirs. Certains types de tirs sont très peu représentés. Par exemple, le dataset contient 1 tir de type "Putback Reverse Dunk Shot", 2 tirs de type "Turnaround Finger Roll Shot" et 3 tirs de type "Running Finger Roll Shot".

          - :blue-background[**shotPoints**] : indique si le joueur tente un tir à 2 ou 3 points.

          - :blue-background[**player**] : prénom et nom du joueur.

          - :blue-background[**year, month**] : mois et année du match. Ces 2 variables sont calculées lors du preprocessing des tirs.
          """
        )

  with st.expander('Voir la liste des variables des dataframes Actions utilisées'):
        st.write(
          """
          - :blue-background[**previousEvent**] : indique l'action qui précède le tir. Cette variable est calculée lors du preprocessing des actions. Dans le dataset des actions, on peut facilement déterminer l'action précédente en utilisant la fonction shift. Pour déterminer si l'action précédente est un lancer-franc réussi ou manqué, on regarde si le score évolue ou non lors de l'action précédente. Pour déterminer si l'action précédente est un rebond offensif ou défensif, on regarde si l'action précédente concerne l'équipe du joueur ou l'équipe adverse. Les autres types d'actions précédentes sont plus précis et ne nécessitent pas de règle supplémentaire. Cette variable est ensuite encodée dans le notebook encoding.ipynb.

          - :blue-background[**lastMin**] : valorisée à 1 pour la dernière minute de jeu de chaque période, et à 0 dans le cas contraire. Elle est calculée dans le notebook encoding.ipynb.

          - :blue-background[**lastPeriod**] : valorisée à 1 pour la dernière période de jeu réglementaire (dernier quart-temps) et pour les prolongations, et à 0 dans le cas contraire. Cette variable est aussi calculée lors de l'encoding.          

          - :blue-background[**playerAverage**] : taux moyen de réussite du joueur sur l'ensemble des tirs précédents et disponibles dans le dataset. Cette variable est aussi calculée lors de l'encoding.

          - :blue-background[**playerShape**] : taux moyen de réussite du joueur sur les 125 tirs précédents. Un joueur tire environ 25 fois par match, la valeur 125 équivaut donc à environ 5 matchs (approximatif, car le nombre de tirs par joueur et par match est très variable). Cette variable est calculée lors de l'encoding.

          - :blue-background[**home**] : indique si le joueur est à domicile ou à l'extérieur. Cette variable est calculée lors du preprocessing des actions.

          - :blue-background[**oppWinPercentage**] : indique le ratio de matchs gagnés (nombre de matchs gagnés / nombre total de matchs) par l'équipe adverse, depuis le début de la saison, à la veille du match. Cette donnée est issue du fichier "ranking.csv" (dataset : NBA games data) et elle est ajoutée lors du merge. Si l'équipe adverse a un ratio élevé de matchs gagnés, il est assez probable qu'elle ait une bonne défense, et donc que le taux de réussite de l'attaquant baisse légèrement.

          - :blue-background[**winPercentage**] : indique le ratio de matchs gagnés par l'équipe du joueur, depuis le début de à la veille du match. Comme pour la variable précédente, cette variable est issue du fichier "ranking.csv" et ajoutée lors du merge. L'impact sur le taux de réussite est moins évident. Si l'équipe a un ratio élevé, les joueurs ont peut-être un meilleur taux de réussite, mais cela restera à vérifier lors de la modélisation.

          """
        )
        
  st.subheader('III. Fusion des datasets pour la modélisation')

  st.markdown(
    """
    A l'issue du preprocessing et du feature engineering, les données sont exportées dans 4 fichiers CSV :

    - le fichier shot_preprocessed
    - le fichier actions_partial.csv contient les tirs dans le jeu issus du site Kaggle
    - le fichier actions_full.csv contient tous les tirs dans le jeu (Kaggle + API)
    - le fichier actions_with_free_throws.csv contient les tirs dans le jeu et les lancers-francs (Kaggle + API)

    Par la suite, c'est principalement le fichier actions_full.csv qui est utilisé et fusionné avec le fichier shots_preprocessed, donc tous les tirs dans le jeu.
    
    Pour les lancers-francs, il y a peu de variables, donc peu d'ajustements possibles, et on se limitera à un calcul du taux de réussite, sans travail supplémentaire de feature engineering ou d'optimisation des hyperparamètres.

    :blue-background[**Les dataframes shot_preprocessed et actions_full sont fusionnés sur le nouvel index crée avec l'identifiant du match et le numéro d'événement dans le match**]

    Les étapes sont les suivantes :

    - Suppression des actions absentes du dataset des tirs (74 actions) et inversement (1 tir)
    - Merge des 2 datasets
    -  Ajout de données issus du dataset "ranking"
    - Renommage et filtrage des colonnes

    Les datasets initiaux représentent un volume de données  important, notamment en ce qui concerne les tirs (4,7 millions de lignes pour la période 1997-2020) et les actions (8,8 millions de lignes pour la période 2003-2019).
    Après le filtrage sur les 20 joueurs étudiés et la fusion, nous obtenons un tableau d'environ 217 000 lignes.    
    """
  )
  
  df = pd.read_csv('_PRE_PRO_DATAS/merged.csv', index_col=0)
  rows, cols = df.shape
  with st.expander("Voir un extrait du jeu de données"):
    st.write(f"Le dataframe contient {rows} lignes et {cols} colonnes.")
    st.dataframe(
      df.head(10),
      column_config={
        'year' : st.column_config.NumberColumn(format='%d'),
        'Player ID' : st.column_config.NumberColumn(format='%d'),
        'Game ID' : st.column_config.NumberColumn(format='%d')
      } 
      )
  
  with st.expander("Voir le dataframe transposé pour lister les variables"):
    st.dataframe(
      df.tail(6).T,
      column_config={
        'year' : st.column_config.NumberColumn(format='%d'),
        'Player ID' : st.column_config.NumberColumn(format='%d'),
        'Game ID' : st.column_config.NumberColumn(format='%d')
      } 
      )
