# Importation des librairies
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.stats import chi2_contingency
import statsmodels.api as sm

import shap

import streamlit as st


#------------------------------------------------------------------------------------------------------------------------------------
## FONCTION D'INITIALISATION ##
#------------------------------------------------------------------------------------------------------------------------------------

@st.cache_data
def init_zones_analytics():
    # Initialisation pour l'analyse par zone :
    # - chargement des données
    # - création des dataframes par joueur et par zone

    # Fichier csv à charger
    file = '../_DATAS/NBA Shot Locations 1997 - 2020.csv'
    
    # Liste des joueurs à conserver
    top20 = ['Giannis Antetokounmpo', 'Stephen Curry', 'LeBron James', 'Kevin Durant', 'Anthony Davis', 'Paul George', 'Kawhi Leonard', 'Damian Lillard', 'Kyrie Irving', 'Jimmy Butler', 'Rudy Gobert', 'Jrue Holiday', 'James Harden', 'DeMar DeRozan', 'Kentavious Caldwell-Pope', 'Khris Middleton', 'Draymond Green', 'Bradley Beal', 'Klay Thompson', 'Jonas Valanciunas']
    top20 = [x.upper() for x in top20]

    # Liste des variables à supprimer
    drop_var = ['Team ID', 'Team Name', 'Season Type', 'Away Team', 'Home Team']

    
    df = top_players_dataframe(file = file, 
                               col_filter = 'Player Name', 
                               list_players = top20, 
                               index = 'Player ID', 
                               drop_var = drop_var)
    
    players_df = df_per_player(df=df)

    zones_var = ['Player Name', 'Action Type', 'Shot Type', 'Shot Zone Basic', 
                 'Shot Zone Area', 'Shot Zone Range', 'Shot Distance', 
                 'X Location', 'Y Location', 'Shot Made Flag']
    
    zones_data = df[zones_var]

    return top20, df, zones_data, players_df


@st.cache_data
def top_players_dataframe(file, col_filter, list_players, index, drop_var):
    # Fonction pour charger un dataframe,
    # le filtrer selon une liste de joueurs,
    # traiter la variable de date,,
    # modifier l'index


    # Chargement du dataset
    df = pd.read_csv(file)

    # Mise en capitale des noms de joueurs dans le dataframe et dans la liste de joueurs Top 20
    df[col_filter] = df[col_filter].str.upper()
    list_players = list(map(str.upper, list_players)) 
    #Construction du dataset sur la liste de joueurs
    df = df[df[col_filter].isin(list_players)]

    # Variable à passer en datetime et création des variables 'year' et 'month'
    var_to_datetime = 'Game Date'
    df[var_to_datetime] = df[var_to_datetime].values.astype(str)
    df[var_to_datetime] = pd.to_datetime(df[var_to_datetime], yearfirst=True)
    df['year'] = df[var_to_datetime].dt.year
    df['month'] = df[var_to_datetime].dt.month

    # Choix de l'index et suppression des variables dans la liste
    df = df.set_index(index)
    #df = df.drop('Unnamed: 0', axis=1)
    df = df.drop(drop_var, axis=1)
    
    return df



def to_datetime(df, var):
    # Fonction pour passer une liste de variable au datetime dans un dataframe
    df[var] = df[var].values.astype(str)
    df[var] = pd.to_datetime(df[var], yearfirst=True)
    return df



def df_per_player(df):
    # Fonction de création d'un dictionnaire comportant un dataframe pour chaque joueur
    print("Création d'un dictionaire comportant un dataframe par joueur...")
    players_df = {}
    for player in df['Player Name'].unique():
        player_df = df[df['Player Name'] == player]
        players_df[player] = player_df
    return players_df


#------------------------------------------------------------------------------------------------------------------------------------
## FONCTIONS DE DATA VISUALISATION ##
#------------------------------------------------------------------------------------------------------------------------------------

@st.cache_data
def plotly_pie_plots(zones_data=None, zone_var='Shot Zone Basic'):
    # Vérification 
    if zones_data is None or zone_var not in zones_data.columns:
        raise ValueError("Les données ou la colonne spécifiée sont invalides.")

    # Compter les occurrences de chaque zone
    shots_per_zone = zones_data[zone_var].value_counts().reset_index()
    shots_per_zone.columns = ['Zone', 'Count']

    colors = ['yellowgreen', 'mistyrose', 'lightsteelblue', 'lightcoral', 'thistle', 'paleturquoise']

    fig = go.Figure(data=[go.Pie(
        labels=shots_per_zone['Zone'],  
        values=shots_per_zone['Count'], 
        hole=0.3,  
        marker=dict(colors=colors, line=dict(color='#000000', width=2)), 
        textinfo='percent+label', 
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title={
            'text': f"Répartition des tirs par {zone_var}", 
            'y': 0.95,  
            'x': 0.5,  
            'xanchor': 'center',
            'yanchor': 'top',   
            'font': {'size': 11, 'family': 'Arial'}
        },
        showlegend=False,  
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,  
            xanchor="left",
            x=1.05  
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=400,  
        width=600,   
    )

    return fig


@st.cache_data
def plotly_boxplot_zones(shot_result=None, x=None, y=None):
    # Vérifications
    if shot_result != 'manqués' and shot_result != 'réussis':
        raise ValueError("Le paramètre shot_result doit être 'manqués' ou 'réussis'.")

    if shot_result == 'manqués':
        color='lightsteelblue'
    else:
        color='lime'
        
    if x is not None and y is not None:
        raise ValueError('Veuillez entrer une valeur pour x ou une pour y, mais pas les deux en même temps.')
    elif x is None and y is None:
        raise ValueError('Veuillez entrer une valeur pour x ou une pour y.')

    # Créer la figure
    fig = go.Figure()

    # Créer le boxplot en fonction de x ou y
    if x is not None:
        fig.add_trace(go.Box(
            x=x,
            #name=f'Tirs {shot_result} (X Loc.)',
            boxmean='sd',
            marker_color=color,
            line_color='saddlebrown',
            fillcolor=color,
            whiskerwidth=0.2
        ))
        title_text = f'Distrib. tirs {shot_result} sur X Loc.'
    else:
        fig.add_trace(go.Box(
            y=y,
            #name=f'Tirs {shot_result} (Y Loc.)',
            boxmean='sd',
            marker_color=color,
            line_color='saddlebrown',
            fillcolor=color,
            whiskerwidth=0.2
        ))
        title_text = f'Distrib. tirs {shot_result} sur Y Loc.'

    # Mise en forme de la figure
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        height=400,
        width=1200,
        plot_bgcolor='white',  # Fond blanc pour la zone de tracé
        paper_bgcolor='white',  # Fond blanc pour l'arrière-plan de la figure
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),  # Grille grise pour l'axe X
        yaxis=dict(showgrid=True, gridcolor='lightgrey'), # Grille grise pour l'axe Y
        title={
            'text': title_text,  # Titre centré
            'y': 0.95,  # Position verticale du titre
            'x': 0.5,   # Position horizontale du titre (centré)
            'xanchor': 'center',  # Ancrage horizontal au centre
            'yanchor': 'top',     # Ancrage vertical en haut
            'font': {'size': 16}  # Taille de la police
        }
    )

    # Afficher le graphique
    return fig


@st.cache_data
def plotly_scatter_global_shotZones(zones_data):
    # Filtrer les tirs à 2 points et à 3 points
    shots_2PT = zones_data[zones_data['Shot Type'] == '2PT Field Goal']
    shots_2PT = shots_2PT.sample(10000)
    
    shots_3PT = zones_data[zones_data['Shot Type'] == '3PT Field Goal']
    shots_3PT = shots_3PT.sample(10000)
    
    zones_data = zones_data.sample(10000)
    zones_df = {'Global': zones_data, 'Tirs à 2 points': shots_2PT, 'Tirs à 3 points': shots_3PT}

    # Créer une figure avec 3 sous-graphiques
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(zones_df.keys()))

    # Boucle sur les DataFrames et ajout des traces
    for i, (key, df) in enumerate(zones_df.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=df['X Location'],
                y=df['Y Location'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=df['Shot Made Flag'],  
                    colorscale=['lightsteelblue', 'limegreen'],
                    showscale=True if i == 1 else False,  # Afficher la colorbar uniquement pour le premier graphique
                    colorbar=dict(title='Shot Made Flag', tickvals=[0, 1], ticktext=['Tir manqué', 'Tir réussi'])
                ),
                name=key,
                hoverinfo='x+y+text',
                hovertext=df['Shot Made Flag'].apply(lambda x: 'Tir réussi' if x == 1 else 'Tir manqué')
            ),
            row=1, col=i
        )

    # Mise en forme de la figure
    fig.update_layout(
        title="Répartition des tirs réussis et manqués sur le terrain",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=600,
        yaxis1=dict(range=[-70, 800]), 
        yaxis2=dict(range=[-70, 800]), 
        yaxis3=dict(range=[-70, 800])
    )



    # Afficher la figure
    return fig

@st.cache_data
def plotly_playerZones(players_df):
    # Nombre de sous-graphiques à afficher
    num_plots = len(players_df)
    
    # Calculer le nombre de lignes et de colonnes en fonction du nombre d'éléments
    cols = 5
    rows = (num_plots // cols) + (1 if num_plots % cols > 0 else 0)

    width = 300 * cols  # Largeur (ajuster par rapport au nombre de colonnes)
    height = 300 * rows
    
    # Créer la figure avec le nombre de sous-graphiques adapté
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=list(players_df.keys())[:num_plots]
    )
    
    for i, (key, df) in enumerate(players_df.items(), start=1):
        ech = 2000
        df = df.sample(ech)
        # Calculer la ligne et la colonne pour chaque sous-graphe
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=df['X Location'],
                y=df['Y Location'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=df['Shot Made Flag'],  
                    colorscale=['lightsteelblue', 'limegreen'],                    
                    #colorbar=dict(title='Shot Made Flag', tickvals=[0, 1], ticktext=['Tir manqué', 'Tir réussi']) if i == 1 else None
                ),
                name=key,
                hoverinfo='x+y+text',
                hovertext=df['Shot Made Flag'].apply(lambda x: 'Tir réussi' if x == 1 else 'Tir manqué'),
            ),
            row=row, col=col
        )

    # Mise en forme de la figure
    fig.update_layout(
        title=f"Répartition des tirs réussis et manqués sur le terrain (échantillons de {ech} tirs)",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,  # Hauteur dynamique
        width=1200
    )
    for i in range(1, num_plots + 1):
        fig.update_layout(**{f'yaxis{i}': dict(range=[-70, 500])})

    # Afficher la figure
    return fig

@st.cache_data
def plot_shotPtsType(shotPtsType):
    # Fonction qui affiche sur une seule ligne une table et un graph en barres empilées
    print("\033[1m" + "Taux de réussite des tirs à 2 points et des tirs à 3 points" + "\033[0;0m")

    shotPtsType = shotPtsType.round(2) 
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].axis('off')
    header_color = 'lightsteelblue'
    
    ax[0].table(cellText=shotPtsType.values, 
                colLabels=shotPtsType.columns,
                rowLabels=shotPtsType.index,
                colColours=[header_color] * len(shotPtsType.columns),
                loc='center',
                cellLoc='center',
                colWidths=[0.2] * len(shotPtsType.columns))
    for (i, j), cell in ax[0].get_children()[0].get_celld().items():
        cell.set_fontsize(8) 
    
    
    colors = ['yellowgreen', 'lightsteelblue']
    ax[1].bar(shotPtsType.index, shotPtsType[0], label="Paniers non marqués", color=colors[1])
    ax[1].bar(shotPtsType.index, shotPtsType[1], label="Paniers marqués", bottom=shotPtsType[0], color=colors[0])
    ax[1].set_title("Répartition des paniers", fontsize=7)
    ax[1].set_ylabel("Taux de réussite", fontsize=6)
    ax[1].tick_params(axis='both', labelsize=6)
    ax[1].legend(fontsize=6)
    
    plt.subplots_adjust(wspace=.4, hspace=.4)
    st.pyplot(fig, use_container_width=False)


@st.cache_data
def shots_split_per_player(players_dict):
    # Répartition des tirs à 2 et 3 points pour chaque joueur, en graphique à barres empilées
    fig, axes = plt.subplots(5,4, figsize=(20,20))
    axes = axes.flatten()
    colors = ['thistle', 'paleturquoise']

    for i, (key, df) in enumerate(players_dict.items()):
        tab = df.groupby('year')['Shot Type'].value_counts(normalize=True)
        tab = tab.unstack(level='Shot Type')
        
        ax = axes[i]
        tab.plot(stacked=True, kind='bar', ax=ax, legend=False, color=colors)

        ax.set_title(key, fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Répartition des tirs à 2 et 3 points pour chaque joueur", fontsize=17, y=1.05)
    st.pyplot(fig, use_container_width=False)



@st.cache_data
def distribution_plots(df, var_to_plot, hue, color1='yellowgreen', color2='lightsteelblue'):
    # Fonction pour affiche les histogrammes avec KDE, courbes ECDF et QQ Plot des variables continues

    # Créer la figure et les axes
    print("\033[1m" + "Distribution des variables" + "\033[0;0m")
    fig, axes = plt.subplots(len(var_to_plot), 3, figsize=(15, len(var_to_plot)*3))

    # Palette personnalisée pour les deux classes de 'Shot Made Flag'
    custom_palette = {0: color2, 1: color1}

    # Parcours de chaque variable dans la liste
    for i, var in enumerate(var_to_plot):
        # Histogramme et KDE
        sns.histplot(data=df, x=var, hue=hue, kde=True, bins=100, ax=axes[i, 0], palette=custom_palette, hue_order=[0, 1])

        # Courbe ECDF
        sns.ecdfplot(data=df, x=var, hue='Shot Made Flag', ax=axes[i, 1], palette=custom_palette, hue_order=[0, 1])

        # QQ Plot
        sm.qqplot(df[var], fit=True, line='45', ax=axes[i, 2])

    # Ajuster l'espacement entre les graphiques
    plt.tight_layout()
    #plt.show()
    st.pyplot(fig, use_container_width=False)






@st.cache_data
def playerZones(players_df):
    # Nuage de points des zones de tirs pour chaque joueur

    print("\033[1m" + "Emplacement des tirs pour chaque joueur" + "\033[0;0m")
    print("Les tirs réussis sont affichés en vert et ceux manqués sont affichés en violet.")
    print("Le fond de court, au-delà du milieu de terrain, n'est pas affiché car cela représente très peu de tirs.")

    fig, axs = plt.subplots(5,4, figsize=(9, 12))
    axs = axs.flatten()
    
    # Boucle sur les DataFrames de players_shot et création d'un scatter pour chacun
    for i, (key, df) in enumerate(players_df.items()):
        # Création d'une cmap pour personnaliser les couleurs des points
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors = ['lightsteelblue', 'limegreen'], N=2)
        
        # Création du scatter plot pour chaque DataFrame
        axs[i].scatter(df['X Location'], df['Y Location'], s=.3, c=df['Shot Made Flag'], cmap=cmap)
        axs[i].set_xlim(-250,250)
        axs[i].set_ylim(-50,470)
        axs[i].tick_params(axis='x', labelsize=9)
        axs[i].tick_params(axis='y', labelsize=9)
        
        # Titre pour chaque scatter avec la clé
        axs[i].set_title(key, fontsize=6)

    plt.legend()
    fig.suptitle("Localisation des tirs pour chaque joueur", fontsize=12, y=1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

@st.cache_data
def success_rate_per_year(players_dict):
    # Taux de réussite au tir par an our chaque joueur dans un graphique en courbes

    fig = go.Figure()
    for key, df in players_dict.items():
        tab = df.groupby('year')['Shot Made Flag'].mean()
        fig.add_trace(go.Scatter(x=tab.index, y=tab.values, mode='lines', name=key))

    fig.update_layout(
        title="Taux de réussite au tir annuel par joueur",
        legend_title="Légende",
        width=1100,
        height=600
    )
    st.plotly_chart(fig, use_container_width=False)

@st.cache_data
def plotly_display_shot_zones(df, zone_var='Shot Zone Basic'):
    df=df.sample(10000)
    # Vérification
    if df is None or zone_var not in df.columns or 'X Location' not in df.columns or 'Y Location' not in df.columns:
        raise ValueError("Le paramètre zone_var doit contenir les colonnes 'Shot Zone Basic', 'Shot Zone Area' ou 'Shot Zone Range'.")

    default_colors = [
      'paleturquoise', 'lightcoral', 'lime','thistle', 'lemonchiffon', 'lightsteelblue', 'peachpuff'
    ]

    unique_zones = df[zone_var].unique()

    dynamic_colors = {}
    for i, zone in enumerate(unique_zones):
        dynamic_colors[zone] = default_colors[i % len(default_colors)] 

    # Créer une figure vide
    fig = go.Figure()

    # Ajouter des traces pour chaque groupe de zones
    for name, group in df.groupby(zone_var):
        color = dynamic_colors[name]
        fig.add_trace(go.Scatter(
            x=group['X Location'],
            y=group['Y Location'],
            mode='markers',  
            marker=dict(size=3,color=color),
            name=name
        ))

        legend=dict(
            title=dict(text=zone_var, font=dict(size=12)),
            font=dict(size=10),
            marker=dict(size=10),
            itemsizing='constant'
        ),

        fig.update_layout(
        title=f"Par {zone_var}",
        xaxis=dict(
            title='X Location', 
        ),
        yaxis=dict(
            title='Y Location',  
        ),
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        width=600, 
        height=500 
    )

    

    # Afficher le graphique
    return(fig)


@st.cache_data
def previous_event_viz(fg):

    # Taux de réussite : moyenne pour les 20 joueurs
    fig, axe = plt.subplots(figsize=(8, 4))
    mean_percent = len(fg[fg.EVENT == 'Field goal made'])/len(fg)
    plt.axhline(y=mean_percent, xmin=0.05, xmax=0.95, color='black', label = "Taux global")

    # Pour l'affichage sous forme de graphique, on ne garde que les événements principaux
    fg1 = fg.copy()
    previous_events = fg1.PREVIOUS_EVENT.value_counts(normalize=True)
    for previous_event in previous_events.tail(11).index:
        fg1.PREVIOUS_EVENT = fg1.PREVIOUS_EVENT.replace(previous_event, 'Other')
    
    # Taux de réussite en fonction de l'événement précédent : moyenne pour les 20 joueurs
    ct = pd.crosstab(fg1.PREVIOUS_EVENT, fg1.EVENT)
    ct.columns = ['FGA', 'FGM']
    ct['FGM%'] = ct.FGM/(ct.FGA + ct.FGM)
    ct = ct.sort_values(by = 'FGM%')
    sns.lineplot(x=ct.index, y=ct['FGM%'], color='blue', marker='o', label="Moyenne pour les 20 joueurs")

    # Taux de réussite en fonction de l'événement précédent, pour LeBron James
    ct_lj = pd.crosstab(fg1[fg1.PLAYER == 'LeBron James'].PREVIOUS_EVENT, fg1[fg1.PLAYER == 'LeBron James'].EVENT)
    ct_lj.columns = ['FGA', 'FGM']
    ct_lj['FGM%'] = ct_lj.FGM/(ct_lj.FGA + ct_lj.FGM)
    sns.lineplot(x=ct_lj.index, y=ct_lj['FGM%'], color='green', marker='o', label="LeBron James")

    # Taux de réussite en fonction de l'événement précédent, pour Rudy Gobert
    ct_rg = pd.crosstab(fg1[fg1.PLAYER == 'Rudy Gobert'].PREVIOUS_EVENT, fg1[fg1.PLAYER == 'Rudy Gobert'].EVENT)
    ct_rg.columns = ['FGA', 'FGM']
    ct_rg['FGM%'] = ct_rg.FGM/(ct_rg.FGA + ct_rg.FGM)
    sns.lineplot(x=ct_rg.index, y=ct_rg['FGM%'], color='orange', marker='o', label="Rudy Gobert")
    
    # Mise en forme du graphique
    axe.lines[2].set_linestyle("--")
    axe.lines[3].set_linestyle("--")
    axe.set_xlabel("Actions qui précèdent les tirs", fontsize=6)
    axe.set_ylabel("Taux de réussite des tirs", fontsize=6)
    axe.tick_params(axis='x', labelsize=6)
    axe.tick_params(axis='y', labelsize=6)
    plt.ylim(bottom=0.4, top=0.75)
    plt.title("Taux de réussite en fonction de l'action précédente", fontsize=7)
    plt.legend(fontsize=6)

    st.pyplot(fig, use_container_width=False)


@st.cache_data
def time_left_viz(fg):

    # Taux de réussite : moyenne pour les 20 joueurs
    fig = plt.figure(figsize=(8, 4))
    mean_percent = len(fg[fg.EVENT == 'Field goal made'])/len(fg)
    plt.axhline(y=mean_percent, color='grey', xmin=11, xmax=0, label = "Taux moyen")

    # Taux de réussite par minute restante : moyenne globale
    ct = pd.crosstab(fg.MINS_LEFT, fg.EVENT)
    ct.columns = ['FGA', 'FGM']
    ct['FGM%'] = round(ct.FGM/(ct.FGA + ct.FGM), 3)
    ct = ct.sort_values(by='MINS_LEFT', ascending=False)
    axe = sns.lineplot(x=ct.index, y=ct['FGM%'], color='black', label = "Moyenne par minute")

    # Taux de réussite par minute restante : moyenne par période
    for period in range(1, 5):
        ct = pd.crosstab(fg[fg.PERIOD == period].MINS_LEFT, fg.EVENT)
        ct.columns = ['FGA', 'FGM']
        ct['FGM%'] = round(ct.FGM/(ct.FGA + ct.FGM), 3)
        sns.lineplot(x=ct.index, y=ct['FGM%'], label = "Période" + str(period))
        axe.lines[period + 1].set_linestyle("--")
        axe.lines[period + 1].set_linewidth(1)
    
    plt.xlim(11, 0)
    plt.ylim(bottom=0.35, top=0.55)
    axe.set_xlabel("Nombre de minutes restantes par période", fontsize=6)
    axe.set_ylabel("Taux de réussite des tirs", fontsize=6)
    axe.tick_params(axis='x', labelsize=6)
    axe.tick_params(axis='y', labelsize=6)
    plt.legend(fontsize=6)
    plt.title("Taux de réussite en fonction du temps restant", fontsize=7)
    st.pyplot(fig, use_container_width=False)

@st.cache_data
def nb_shots_per_player():
    df = pd.read_csv('../_DATAS/games_details.csv', low_memory=False)
    df['PLAYER_NAME'] = df['PLAYER_NAME'].str.upper()
    top20 = ['Giannis Antetokounmpo', 'Stephen Curry', 'LeBron James', 'Kevin Durant', 'Anthony Davis', 
             'Paul George', 'Kawhi Leonard', 'Damian Lillard', 'Kyrie Irving', 'Jimmy Butler', 
             'Rudy Gobert', 'Jrue Holiday', 'James Harden', 'DeMar DeRozan', 'Kentavious Caldwell-Pope', 
             'Khris Middleton', 'Draymond Green', 'Bradley Beal', 'Klay Thompson', 'Jonas Valanciunas']
    top20 = [x.upper() for x in top20]
    df = df[df['PLAYER_NAME'].isin(top20)]
        
    # Calcul des tirs à 2 points tentés (attempted) et réussis (made)
    df['FG2A'] = df['FGA'] - df['FG3A']
    df['FG2M'] = df['FGM'] - df['FG3M']

    # Nombre de tirs par joueur : (tentés, réussis ; 2 points, 3 points, total)
    functions = {'FG2A': 'sum', 'FG2M': 'sum', 'FG3A': 'sum', 'FG3M': 'sum', 'FGA': 'sum', 'FGM': 'sum'}
    shoots_per_player = pd.DataFrame(df.groupby('PLAYER_NAME').agg(functions))
    shoots_per_player = shoots_per_player.sort_values(by='FGA', ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    color_2pts = "paleturquoise"  
    color_3pts = "lightcoral"
    colors = ['yellowgreen', 'mistyrose', 'lightsteelblue', 'lightcoral', 'thistle', 'paleturquoise']


    # Graphique des tirs tentés
    ax[0].bar(shoots_per_player.index, shoots_per_player.FG2A, label='Tirs à 2 points', color=color_2pts)
    ax[0].bar(shoots_per_player.index, shoots_per_player.FG3A, bottom=shoots_per_player.FG2A, label='Tirs à 3 points', color=color_3pts)
    ax[0].set_xticks(range(len(shoots_per_player.index)))
    ax[0].set_xticklabels(shoots_per_player.index, rotation=90)
    ax[0].set_ylim(top=35000)
    ax[0].set_ylabel('Tirs tentés')
    ax[0].set_title('Nombre total de tirs tentés par joueur')
    ax[0].legend()

    # Graphique des tirs réussis
    ax[1].bar(shoots_per_player.index, shoots_per_player.FG2M, label='Tirs à 2 points', color=color_2pts)
    ax[1].bar(shoots_per_player.index, shoots_per_player.FG3M, bottom=shoots_per_player.FG2M, label='Tirs à 3 points', color=color_3pts)
    ax[1].set_xticks(range(len(shoots_per_player.index)))
    ax[1].set_xticklabels(shoots_per_player.index, rotation=90)
    ax[1].set_ylim(top=35000)
    ax[1].set_ylabel('Tirs réussis')
    ax[1].set_title('Nombre total de tirs réussis par joueur')
    ax[1].legend()

    # Affichage du graphique dans Streamlit
    st.pyplot(fig, use_container_width=True)

   


#------------------------------------------------------------------------------------------------------------------------------------
## FONCTIONS STATISTIQUES ##
#------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def display_contingency(df, columns_list, target):
    # Affichage de tables de contingence avec crosstab
    for column in columns_list:
        print('-----------------------------------------------------------------------------------------------------------------------------')
        print("\033[1m" + column + "\033[0;0m")
        print('-----------------------------------------------------------------------------------------------------------------------------')

        contingency = pd.crosstab(df[target], df[column], normalize=1)
        st.dataframe(contingency)


def cramers_v(chi2, n, k, r):
    # Calcul du coefficient de Cramer
    return round(np.sqrt(chi2 / (n * min(k - 1, r - 1))),2)

@st.cache_data
def chi2_testing(df, var_list, target):
    # Test statistique Khi2

    for var in var_list:
        contingency = pd.crosstab(df[var], df[target])
        stat, p, d, e = chi2_contingency(contingency)
        n = contingency.sum().sum()  # Nombre total d'observations
        k = contingency.shape[0] # Nombre de lignes (modalités de la 1ère variable)
        r = contingency.shape[1]
        V_cramer = cramers_v(stat, n, k, r)
        st.write(f"**Test Khi2 pour la variable {var}**")
        st.write(f"- **P-value**: {p:.4f}")
        st.write(f"- **V de Cramer**: {V_cramer:.4f}")
        st.write("---")
    


def chi2_testing_results(df, var_list, target):
    # Fonction de stockage des résultats du Khi2

    results = []
    for zone in var_list:
        contingency = pd.crosstab(df[zone], df[target])
        stat, p, dof, expt_f = chi2_contingency(contingency)
        n = contingency.sum().sum()  # Nombre total d'observations
        k = contingency.shape[0]  # Nombre de lignes (modalités de la 1ère variable)
        r = contingency.shape[1]
        V_cramer = cramers_v(stat, n, k, r)
        results.append([zone, p, V_cramer])
    return results

def df_chi2_building(players_dict, var_list, target):
    # Construction d'un dataframe avec les résultats des Chi2

    # Dataframe vide
    zones_cramer_per_player = pd.DataFrame(columns=['Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range'])
    # Chi2 pour chaque joueur et ajout des résultats dans le dataframe
    for (key, df) in players_dict.items():   
        results = chi2_testing_results(df=df, var_list=var_list, target=target)
        zones_cramer_per_player.loc[key, [results[0][0]]] = results[0][2]
        zones_cramer_per_player.loc[key, [results[1][0]]] = results[1][2]
        zones_cramer_per_player.loc[key, [results[2][0]]] = results[2][2]
    
    fig = go.Figure()
    for col in zones_cramer_per_player.columns:
        fig.add_trace(go.Bar(
            x=zones_cramer_per_player.index,
            y=zones_cramer_per_player[col],
            name=col
        ))
    fig.update_layout(
        barmode='group',  #  =>pour que les barres soient côte à côte
        title='Coefficient V Cramer du Khi2 par joueur et par variable de zones de jeu',
        legend_title='Séries',
        width=1100,
        height=500
    )
    st.plotly_chart(fig, use_container_width=False)

@st.cache_data
def heatmap(df):
    num = df.select_dtypes(np.number)
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype='bool'))
    fig = plt.figure(figsize=(15,6))
    sns.heatmap(data=corr, mask=mask, annot=True, center=0, fmt='.2f', cmap='viridis')
    st.pyplot(fig, use_container_width=True)

    


#------------------------------------------------------------------------------------------------------------------------------------
## FONCTIONS POUR PRÉPARER LES DONNÉES POUR LE MACHINE LEARNING ET LE DEEP LEARNING ##
#------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def setting_scaling_dataframes(df):
    # Fonction pour standardiser et créer les jeux de données d'entraînement et de test
    
    X = df.drop(['target'], axis=1)
    y = df.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
    cols_to_scale = ['Xloc', 'Yloc', 'shotDistance', 'shotPoints']
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    feature_names = X_test.columns.tolist()
    
    print("Format des jeux d'entrainement et de test")
    print(" - X_train : ", X_train.shape)
    print(" - X_test : ", X_test.shape)
    return X, y, X_train, X_test, y_train, y_test, feature_names

@st.cache_data
def convert_to_numpy_array(X_train, X_test):
    # Fonction pour transformer les jeux de données en tableaux Numpy pour le deep Learning
    X_train_array = X_train.to_numpy()
    X_train_array = X_train_array.astype(np.float32)
    X_test_array = X_test.to_numpy()
    X_test_array = X_test_array.astype(np.float32)
    return X_train_array, X_test_array


@st.cache_data
def subsetting(model, X_train, X_test):
    # Fonction pour créer des échantillons
    X_train_subset = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    X_test_subset = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
    preds_subset = model.predict(X_test_subset)
    print("Format de l'échantillon d'entraînement : ", X_train_subset.shape)
    print("Format de l'échantillon de test : ", X_test_subset.shape)
    print("Format des prédictions sur l'échantillon : ", X_test_subset.shape)
    print('------------------------------------------------------------------------------------------------------------------------')
    return(X_train_subset, X_test_subset, preds_subset)


#------------------------------------------------------------------------------------------------------------------------------------
## FONCTIONS POUR SHAP ##
#------------------------------------------------------------------------------------------------------------------------------------

shap.initjs()

@st.cache_data
def SHAP_explanations(_model, X_train, X_test, preds, feature_names, explainer_type='TreeExplainer'):
    # Fonction pour créer les valeurs SHAP et les organiser adans un objet facilement manipulable

    # Plusieurs possibilité pour l'explainer de SHAP
    if explainer_type == 'DeepExplainer':
        print("Utilisation de DeepExplainer...")
        X_train = X_train.to_numpy()
        X_train = X_train.astype(np.float32)
        X_test = X_test.to_numpy()
        X_test= X_test.astype(np.float32)        
        explainer = shap.DeepExplainer(_model, X_train)
    
    elif explainer_type == 'KernelExplainer':
        print("Utilisation de KernelExplainer...")
        X_train=X_train.sample(5)
        explainer = shap.KernelExplainer(_model.predict, X_train, n_jobs=-1, link="logit")

    elif explainer_type == 'TreeExplainer':
        print("Utilisation de TreeExplainer...")
        explainer = shap.TreeExplainer(_model)
    else:
        raise ValueError(f"Explainer type '{explainer_type}' non reconnu. Utiliser 'DeepExplainer', 'KernelExplainer' ou 'TreeExplainer'.")

    explanation = explainer(X_test)
    
    # Exploration de l'objet 'explanation' et extraction des datas
    shap_values, mean_base_value = SHAP_explanation_checking(explanation=explanation)
    print('------------------------------------------------------------------------------------------------------------------------')
    
    print("Création des valeurs manquantes")

    if 'shap_values' not in locals() or shap_values is None:  # Vérifie si shap_values existe déjà
      shap_values = explanation.values.squeeze()  # squeeze() élimine les dimensions de taille égale à 1 ; nous voulons mettre shap_values à 2 dimensions pour les plots
      print("Format de shap_values : ", shap_values.shape)  
    else:
      print("shap_values existe déjà.")
      
    if 'mean_base_value' not in locals() or mean_base_value is None:  # Vérifie si base_value existe déjà
      mean_base_value = np.mean(preds, axis=0) # moyenne des prédictions du modèle sur l'échantillon
      mean_base_value  = round(mean_base_value[0],2)
      print("Moyenne des 'base_values', soit la moyenne des prédictions du modèle DNN :", mean_base_value)
    else:
      print("mean_base_value existe déjà.")
      

    print('------------------------------------------------------------------------------------------------------------------------')
    
    # Reconstitution de 'explanation' dans un nouvel objet pour le manipuler facilement
    print("\nCréation d'un objet avec les valeurs calculées et les feature_names.", end = '\n\n') 
    explanation_obj = shap.Explanation(values=shap_values, 
                                       base_values=mean_base_value, 
                                       data=X_test, 
                                       feature_names=feature_names)
    
    print("Vérification du nouvel objet explanation_obj :", type(explanation_obj), end = '\n\n')
    print("Format de explanation : ", explanation_obj.shape, end = '\n\n')
    print("Format des SHAP values :", explanation_obj.values.shape, end = '\n\n')
    print("Moyenne des Base Values : ", explanation_obj.base_values, end = '\n\n')
    print("Nombre de Feature Names: ", len(explanation_obj.feature_names), end = '\n\n')
    print('------------------------------------------------------------------------------------------------------------------------')

    return explanation, explanation_obj, shap_values, mean_base_value


def SHAP_explanation_checking(explanation):
    # Fonction pour vérifier les valeurs dans l'objet SHAP 'explanation'

    missing_attributes = []
    print("Type de l'objet 'explanation' :", type(explanation), end = '\n\n')
    print("Format de l'explanation' :", explanation.shape, end = '\n\n')

    # Vérification de l'attribut `values`
    if hasattr(explanation, 'values') and explanation.values is not None:
        shap_values = explanation.values.squeeze() 
        print("Format des SHAP values :", shap_values.shape, end = '\n\n')
    else:
        shap_values = None
        missing_attributes.append('values')
        print("Format des SHAP values : None", end = '\n\n')
    
    # Vérification de l'attribut `base_values`
    if hasattr(explanation, 'base_values') and explanation.base_values is not None:
        mean_base_value = round(explanation.base_values.mean(),3)
        print("Moyenne des 'base_values' : ", mean_base_value, end = '\n\n')
    else:
        mean_base_value = None
        missing_attributes.append('base_values')
        print("'base_values' : None", end = '\n\n')
    
    # Vérification de l'attribut `feature_names`
    if hasattr(explanation, 'feature_names') and explanation.feature_names is not None:
        print("Nombre de 'feature_names' :", len(explanation.feature_names), end = '\n\n')
    else:
        missing_attributes.append('feature_names')
        print("'feature_names' : None", end = '\n\n')
    
    if missing_attributes:
        msg = f"Attributs manquants : {', '.join(missing_attributes)}"
    else:
        msg = "Les attributs dont nous avons besoin sont présents."
    
    print(msg)
    return shap_values, mean_base_value


def SHAP_reliability(model_name, shap_values, base_value, preds):
    # Fonction pour vérifier que SHAP explique correctement les prédictions du modèle

    print(f"Prédiction moyenne du modèle {model_name} :", base_value, end='\n\n')
    shap_reconstructed_preds = shap_values.sum(axis=1) + base_value # => prédiction reconstituée par SHAP en ajoutant la somme des shap_values à la moyenne des prédictions
    
    #Vérification de la reconstruction SHAP
    ecart_moyen = np.abs(shap_reconstructed_preds - preds).mean().flatten()
    ecart_max = np.abs(shap_reconstructed_preds - preds).max().flatten()
    
    print("Ecart moyen entre les prédictions reconstituées par SHAP et les prédictions du modèle : ", round(ecart_moyen[0],2), end='\n\n')
    print("Ecart maximal entre les prédictions reconstituées par SHAP et les prédictions du modèle : ", round(ecart_max[0],2), end='\n\n')

@st.cache_data
def SHAP_summary_plot(_explanation, shap_values, X_test, feature_names, max_display=10, title="Features importances"):
    fig, ax = plt.subplots(figsize=(5, 3))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=max_display, show=False)
    plt.title(title)
    return fig

@st.cache_data
def decisionWaterfall(_explanation, event_index=1, max_display=10):
    # Fonction qui affiche la waterfall qui explique pour un évènement la classification du modèle
    title=f'Analyse du tir à l\'index {event_index}'
    fig = plt.figure()
    shap.plots.waterfall(_explanation[event_index], max_display=max_display, show=False)
    plt.title(title)
    return fig




#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#
## ARCHIVES ===> FONCTIONS INTIALES AVEC MATPLOTLIB  ET SEABORN
## LA PLUPART ONT ÉTÉ REFAITES AVEC PLOTLY
#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

@st.cache_data
def pie_plots(zones, titles):
    # Afichage de graphique ede type pie plot
    title_font = {
        'fontsize': 12,
        'fontweight': 'bold',
    }
    
    colors = ['yellowgreen', 'mistyrose', 'lightsteelblue', 'lightcoral', 'thistle', 'paleturquoise']
    
    print("\033[1m" + "Répartition des tirs par zones" + "\033[0;0m")
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    
    for i, (zone, title) in enumerate(zip(zones, titles)):
        axes[i].pie(zone.values, labels=zone.index, autopct='%1.1f%%', colors=colors, normalize=True)
        axes[i].set_title(title, fontdict=title_font)
    
    plt.subplots_adjust(wspace=0.7, hspace=.8)
    st.pyplot(fig)



@st.cache_data
def boxplot_zones(missed, scored):
    # Visualisation des zones des tirs manqués et réussis en boxplot
    print("\033[1m" + "Distribution des tirs en largeur et en longueur" + "\033[0;0m")
    print("Une unité représente 0.1 pied, soit 3 centimètres")

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    title_font={
        'fontsize': 5
    }
    
    sns.boxplot(ax=axes[0], data=missed['X Location'], orient='h',
                boxprops = {'color' :'lightsteelblue'},
                whiskerprops = {'color' : 'saddlebrown'},
                capprops = {'color': "lightcoral"},
                flierprops = {'marker' : 'o', 'color' : 'red', 'markersize' : 2},
                medianprops = {'color': "white", 'linewidth' : 2})
    axes[0].set_title('Distrib. tirs manqués sur la largeur X', fontdict=title_font)
    axes[0].tick_params(axis='both', labelsize=6)
    axes[0].set_xlabel('X Location', fontsize=7)  
    axes[0].set_ylabel('Y Location', fontsize=7) 
    
    sns.boxplot(ax=axes[1], data=missed['Y Location'], orient='v',
                boxprops = {'color' :'lightsteelblue'},
                whiskerprops = {'color' : 'saddlebrown'},
                capprops = {'color': "lightcoral"},
                flierprops = {'marker' : 'o', 'color' : 'red', 'markersize' : 2},
                medianprops = {'color': "white", 'linewidth' : 2})
    axes[1].set_title('Distrib. tirs manqués sur la longueur Y', fontdict=title_font)
    axes[1].tick_params(axis='both', labelsize=6)
    axes[1].set_xlabel('X Location', fontsize=7)  
    axes[1].set_ylabel('Y Location', fontsize=7)
    
    sns.boxplot(ax=axes[2], data=scored['X Location'], orient='h',
                boxprops = {'color' :'yellowgreen'},
                whiskerprops = {'color' : 'saddlebrown'},
                capprops = {'color': "yellowgreen"},
                flierprops = {'marker' : 'o', 'color' : 'red', 'markersize' : 2},
                medianprops = {'color': "white", 'linewidth' : 2})
    axes[2].set_title('Distrib. tirs réussis sur la largeur X', fontdict=title_font)
    axes[2].tick_params(axis='both', labelsize=6)
    axes[2].set_xlabel('X Location', fontsize=7)  
    axes[2].set_ylabel('Y Location', fontsize=7)
    
    sns.boxplot(ax=axes[3], data=scored['Y Location'], orient='v',
                boxprops = {'color' :'yellowgreen'},
                whiskerprops = {'color' : 'saddlebrown'},
                capprops = {'color': "yellowgreen"},
                flierprops = {'marker' : 'o', 'color' : 'red', 'markersize' : 2},
                medianprops = {'color': "white", 'linewidth' : 2})
    axes[3].set_title('Distrib. tirs réussis sur la longueur Y', fontdict=title_font)
    axes[3].tick_params(axis='both', labelsize=6)
    axes[3].set_xlabel('X Location', fontsize=7)  
    axes[3].set_ylabel('Y Location', fontsize=7)
    
    plt.subplots_adjust(wspace=.6, hspace=.4)
    st.pyplot(fig, use_container_width=False)


@st.cache_data
def display_shot_zones(df):

    legend_params = {
        'markerscale': 2.5,
        'fontsize': 5,
        'title_fontproperties': {'weight': 'bold', 'size': '5'}
    }

    fig = plt.figure(figsize=(5, 2.5))
    plt.subplot(131)
    for name, group in df.groupby('Shot Zone Basic'):
        plt.plot(group['X Location'], group['Y Location'], marker='o', linestyle='', ms=1, label=name)

    plt.legend(**legend_params, title="Shot zone basic")
    plt.xlabel('X Location', fontsize=5)
    plt.ylabel('Y Location', fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(132)
    for name, group in df.groupby('Shot Zone Area'):
        plt.plot(group['X Location'], group['Y Location'], marker='o', linestyle='', ms=1, label=name)
    plt.legend(**legend_params, title="Shot zone area")
    plt.xlabel('X Location', fontsize=5)
    plt.ylabel('Y Location', fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.subplot(133)
    for name, group in df.groupby('Shot Zone Range'):
        plt.plot(group['X Location'], group['Y Location'], marker='o', linestyle='', ms=1, label=name)
    plt.legend(**legend_params, title="Shot zone range")
    plt.xlabel('X Location', fontsize=5)
    plt.ylabel('Y Location', fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    st.pyplot(fig, use_container_width=False)

@st.cache_data
def scatter_global_shotZones(zones_df):
    # Nuage de points pour les zones globales

    print("\033[1m" + "Répartition des tirs réussis et manqués sur le terrain" + "\033[0;0m")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes = axes.flatten()
    
    # Boucle sur les DataFrames de shots_df et création d'un scatter pour chacun
    scatters = []
    for axe, (key, df) in zip(axes, zones_df.items()):
        # Création d'une cmap pour personnaliser les couleurs des points
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors = ['lightsteelblue', 'limegreen'], N=2)
        
        # Création du scatter plot pour chaque DataFrame
        scatter = axe.scatter(df['X Location'], df['Y Location'], s=.3, c=df['Shot Made Flag'], cmap=cmap)
        axe.set_title(key, fontsize=7)
        axe.set_ylim(-50,900)
        axe.set_xlabel('X Location', fontsize=5)
        axe.set_ylabel('Y Location', fontsize=5)
        axe.tick_params(axis='both', labelsize=6) 
        scatters.append(scatter)
    fig.tight_layout(pad=2)

        # Création d'une colorbar partagée par les 3 scatter
    cbar = fig.colorbar(scatters[0], ax=axes, orientation='vertical')
    cbar.set_ticks([0, 1]) 
    cbar.set_ticklabels(['Tir manqué', 'Tir réussi'])
    cbar.ax.tick_params(labelsize=6) 

    st.pyplot(fig, use_container_width=False)