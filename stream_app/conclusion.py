import streamlit as st


def main():
  st.title("Conclusion")
  st.markdown(
    """
Ce projet nous a beaucoup plu. C'était notre premier choix. Nous avons parfois tatonné, mais au final, nous avons pu obtenir des résultats que nous trouvons intéressants et nous avons mis en pratique une bonne partie des connaissances acquises tout au long de la formation.

##### Plusieurs bonnes surprises

- Les jeux de données sur lesquels nous avons travaillé ont nécessité peu de nettoyage.
- Nous avons découvert à quel point le classifieur XGBoost est performant. En quelques secondes, il donne des résultats un peu meilleurs que le classifieur Random Forest et bien meilleurs que la régression logistique et l'arbre de décision, en tout cas pour notre usage.
- Les temps de traitement ont été meilleurs que ce que nous pensions au regard du volume de données. Par exemple, le preprocessing des 8 millions d'actions ne prend que 5 minutes et le calcul du taux de réussite moyen de chaque joueur sur l'ensemble de leurs tirs précédents prend moins d'une seconde.
- L'API nba_api est simple à utiliser. Nous avons facilement pu récupérer les 1307 matchs absents du dataset des actions.
- Le module SHAP est d'une grande aide pour l'interprétation des résultats (avec toutefois des temps de calcul assez longs notamment pour le modèle DNN).

##### Autres enseignements

- L'exploration des données peut prendre du temps. En particulier, le dataset des actions a nécessité du temps pour bien comprendre son contenu, car plusieurs colonnes ne contiennent que des codes. Par ailleurs, avec le nombre important de variables disponibles, il n'est pas évident d'identifier toutes les variables pertinentes et complètes.
- Dans notre cas, plusieurs variables ont assez peu d'influence sur le taux de réussite. Ce sont surtout les variables présentes dans le dataset des tirs qui influent sur le taux de réussite. Cela peut sembler parfois un peu frustrant, mais comme tout projet, certaines actions ont parfois moins d'effet que d'autres.
- Le travail sur les données (exploration, visualisation, feature engineering) est primordial pour avoir les données les plus complètes et les plus propres possibles. Le paramétrage des hyperparamètres des classifieurs permet d'améliorer un peu les scores, mais **ce sont avant tout les données en entrée qui influent sur les résultats**. Plus on arrive à identifier ou calculer des variables pertinentes, meilleurs sont les résultats.

##### Limite du projet

De notre point de vue, la principale limite du projet est que des données importantes ne sont pas en consultation libre. En particulier, on peut citer les 4 variables suivantes :
- La distance entre le tireur et le dernier défenseur (closest defender)
- Le temps restant avant le délai de 24 secondes autorisé pour une action offensive (shot clock range)
- Le nombre de dribbles effectués par le joueur avant de tirer (dribbles)
- La durée de possession de la balle par le joueur avant de tirer (touch time range)

Ces données sont disponibles pour chaque joueur, mais pas pour chaque tir. Elles influent fortement sur le taux de réussite, comme on peut le voir, par exemple pour [Giannis Antetokounmpo](https://www.nba.com/stats/player/203507/shots-dash?Season=2018-19&PerMode=Totals).

##### Le mot de la fin !

Même si ce projet ne permettra pas de fournir des enseignements aux équipes NBA - nous n'en sommes pas encore là, et c'est normal :) - nous le recommandons fortement pour de futurs apprenants, car le sujet est intéressant, le volume de données à manipuler est conséquent et il permet de tester les différents classifieurs et de comprendre les résultats en utilisant le module SHAP. Dans la fiche projet, il nous semblerait intéressant de mentionner l'API nba_api, qui permet d'obtenir des données plus récentes que celles sur lesquelles nous avons travaillé.
  """
  )
