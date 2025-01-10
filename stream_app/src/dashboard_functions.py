# Importation des librairies
import streamlit as st

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from matplotlib import cm
from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap

sns.set_style('white')


# Initalisation d'un dataframe retravaillé pour le dashboard
@st.cache_data
def set_dataframe():
    shots = pd.read_csv('_PRE_PRO_DATAS/shots_pre-processed.csv')
    actions = pd.read_csv('_PRE_PRO_DATAS/actions_full.csv')
    # Suppression des actions absentes du dataset des tirs
    unmatch = actions[~actions.index.isin(shots.index)]
    actions_filtered = actions[~actions.index.isin(unmatch.index)]
    
    # Suppression des tirs absents du dataset des actions
    unmatch = shots[~shots.index.isin(actions.index)]
    shots_filtered = pd.DataFrame()
    if len(unmatch) < 1000: # on supprime les tirs seulement s'il y en a peu à supprimer = si le dataset des actions est complet
        shots_filtered = shots[~shots.index.isin(unmatch.index)]
        
    
    df = shots.merge(actions_filtered, left_index=True, right_index=True, how='inner')
    df.index.name = None
    
    rankings = pd.read_csv('_DATAS/ranking.csv')
    
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
                 'GAME_ID_x' : 'gameID',
                 'Shot Made Flag' : 'target'}
    
    df = df.rename(columns=var_names).filter(items=list(var_names.values()))

    conditions = [
    df['shotType'].str.upper().str.contains('DUNK'),    
    df['shotType'].str.upper().str.contains('PULLUP'),
    df['shotType'].str.upper().str.contains('LAYUP'), 
    df['shotType'].str.upper().str.contains('JUMP'),
    df['shotType'].str.upper().str.contains('DRIVING')
    ]

    values = ['Dunk', 'Pullup', 'Layup', 'Jump', 'Driving']
    df['shotLabel'] = np.select(conditions, values, default='Other')

    return df


def draw_plotly_court(fig, fig_width=600, margins=10):
    
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5,
                    start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)): path += f'L{x[k]}, {y[k]}'
        if closed: path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084
    three_line_col, main_line_col = "#000000", "#000000"

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False,
                   showline=False, ticks='', showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False,
                   showline=False, ticks='', showticklabels=False),
        shapes=[
            dict(type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                 line=dict(color=main_line_col, width=1)),
            dict(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                 line=dict(color=main_line_col, width=1)),
            dict(type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color=main_line_col, width=1)),
            dict(type="path", path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                 line=dict(color=three_line_col, width=1), layer='below'),
            dict(type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                 line=dict(color=three_line_col, width=1), layer='below'),
            dict(type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                 line=dict(color=three_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
        ]
    )
    return True
  

@st.cache_data
def plot_target_repartition(df):
    count = df.target.value_counts()
    # Graph à barres empilées pour les tirs
    fig = go.Figure()
    # Barre pour la classe 0
    fig.add_trace(go.Bar(
        y=[''], 
        x=[count.get(0, 0)], 
        orientation='h',
        name='Tirs manqués',
        marker=dict(color='lightsteelblue')
    ))
    
    # Barre pour la classe 1
    fig.add_trace(go.Bar(
        y=[''],
        x=[count.get(1, 0)],
        orientation='h',
        name='Tirs réussis',
        marker=dict(color='forestgreen')
    ))
    
    fig.update_layout(
        barmode='stack', 
        showlegend=True,
        template='plotly_dark',
        width=1000,
        height=100,
        xaxis=dict(showticklabels=False),
        margin=dict(l=25, r=25, t=25, b=25)
    )

    fig.add_annotation(
    x=len(df),
    y=0,
    text=f'Total des tirs: {len(df)}',
    showarrow=False,
    font=dict(size=11, color="white"),
    align='left', 
    bgcolor='black',
    borderpad=4
    )
    
    return fig


@st.cache_data
def player_shots_stats_per_game(df, player_name):
    # Vérifier que le joueur existe
    player_name = player_name.upper()
    if player_name not in df.player.unique():
        print(f"Le joueur {player_name} n'existe pas dans les données.")
        return

    df_player = df[df.player == player_name]
    df_successful_shots = df_player[df_player['target'] == 1]
    
    # Calculer le nombre de tirs réussis par match par saison
    shots_per_game_season = df_player.groupby('year')['gameID'].value_counts().reset_index(name='shots_per_game').sort_values(by="gameID")
    
    # Calculer le nombre de points par match par saison en tenant compte uniquement des tirs réussis
    points_per_game_season = df_successful_shots.groupby(['year', 'gameID'])['shotPoints'].sum().reset_index(name='points_per_game').sort_values(by="gameID")
    
    # Fusionner les deux résultats pour avoir à la fois le nombre de tirs et le nombre de points par match
    result = pd.merge(shots_per_game_season, points_per_game_season, on=['year', 'gameID'])
    
    # Calculer le nombre moyen de tirs par match par saison
    avg_shots_per_game_per_season = result.groupby('year')['shots_per_game'].mean().reset_index(name='avg_shots_per_game')
    
    # Calculer le nombre moyen de points par match par saison
    avg_points_per_game_per_season = result.groupby('year')['points_per_game'].mean().reset_index(name='avg_points_per_game')
    
    # Fusionner les deux résultats pour obtenir à la fois le nombre moyen de tirs et le nombre moyen de points par match pour chaque saison
    final_result = pd.merge(avg_shots_per_game_per_season, avg_points_per_game_per_season, on='year')
    y_min = min(final_result['avg_shots_per_game'].min(), final_result['avg_points_per_game'].min())
    y_max = max(final_result['avg_shots_per_game'].max(), final_result['avg_points_per_game'].max())
    
    
    fig = go.Figure()
    # Nombre moyen de tirs par match en barres
    fig.add_trace(go.Bar(
        x=final_result.year,
        y=final_result['avg_shots_per_game'],
        name="Moy. de tirs par match",
        yaxis='y1',  # Axe des ordonnées de gauche
        marker_color='white'
    ))

    # Courbe des points
    fig.add_trace(go.Scatter(
        x=final_result.year,
        y=final_result['avg_points_per_game'],
        mode='lines+markers',
        yaxis='y2',
        name='Moy. de points marqués par match', 
        line=dict(color='forestgreen')
    ))

    fig.update_layout(
        title="Moy. des tirs dans le jeu et points par match par année (hors lancer-francs)",
        xaxis=dict(
            tickangle=-45,
            showgrid=False,  
            zeroline=False, 
            showline=False 
        ),
        yaxis=dict(
            range=[0, y_max + 1],
            titlefont=dict(color='black'),
            tickfont=dict(color='black'),
            showgrid=False,  
            zeroline=False, 
            showline=False
        ),
        yaxis2=dict(
            range=[0, y_max + 1],
            titlefont=dict(color='forestgreen'),
            tickfont=dict(color='forestgreen'),
            overlaying='y',
            side='right',
            showgrid=False,  
            zeroline=False, 
            showline=False 
        ),
        barmode='group', 
        template='plotly_dark',
        width=1000,
        height=500,
        legend=dict(
        orientation='h',  
        yanchor='bottom',
        y=1, 
        xanchor='center',
        x=0.5
        )
    )

    # Afficher le graphique
    return fig

@st.cache_data
def histo_horiz(df):
    df_sorted = df.sort_values(by=df.columns[0], ascending=True)
    df_sorted['total'] = df_sorted.iloc[:, 0]

    fig = go.Figure(go.Bar(
    x=df_sorted['total'],
    y=df_sorted.index,
    orientation='h',  
    marker=dict(color='steelblue', line=dict(width=0)),
    text=df_sorted['total'],  
    textposition='inside', 
    textfont=dict(size=12, color='white'),
    insidetextanchor='end',
    ))
    
    # Ajouter les détails du graphique
    fig.update_layout(
        title='Répartition des types de tirs',
        title_x=0.5,  # Centrer le titre horizontalement
        title_xanchor='center',  # Assurer que le titre est centré
        template='plotly_dark',  
        width=1000,
        height=500,
        showlegend=False,
        xaxis=dict(
            showgrid=False,  
            zeroline=False, 
            showline=False ,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False 
        ),
        margin=dict(l=150),  
    )
    
    # Afficher le graphique
    return fig

@st.cache_data
def previous_event_viz(df, player="Lebron James"):     
    
    # Taux de réussite global

    mean_percent = df.target.mean()
    
    # Copie du DataFrame pour éviter de modifier le df initial
    df1 = df.copy()

    # Regrouper les événements peu fréquents sous 'Other'
    previous_events = df1.previousEvent.value_counts()
    for event in previous_events.tail(8).index: 
        df1['previousEvent'] = df1['previousEvent'].replace(event, 'Other')

    # Calcul du taux de réussite par événement précédent pour tous les joueurs 
    ct = pd.crosstab(df1.previousEvent, df1.target)
    
    ct.columns = ['Field Goals Attempts', 'Field Goals Made']  # FGA = Field Goal Attempts, FGM = Field Goals Made
    ct['successRate_global'] = ct['Field Goals Made'] / (ct['Field Goals Attempts'] + ct['Field Goals Made'])  # Calcul du taux de  global
    ct = ct.sort_values(by='successRate_global')

    # Calcul du taux de réussite par événement précédent pour le joueur spécifié
    player = player.upper()
    df_player = df1[df1.player.str.upper() == player]
    ct_player = pd.crosstab(df_player.previousEvent, df_player.target)
    ct_player.columns = ['Field Goals Attempts', 'Field Goals Made'] 
    ct_player['successRate_player'] = ct_player['Field Goals Made'] / (ct_player['Field Goals Attempts'] + ct_player['Field Goals Made'])

    # Union des datas
    ct = ct.merge(ct_player[['successRate_player']], how='left', left_index=True, right_index=True)

    # Courbe pour tous les joueurs
    #sns.lineplot(x=ct.index, y=ct['successRate'], color='blue', marker='o', linestyle='--', label="Moyenne pour les 20 joueurs sur la période 2003-2020")
    # Graphique pour la moyenne de tous les joueurs (ligne bleue)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ct.index,
        y=ct['successRate_global'],
        mode='lines+markers',
        name="Moyenne pour les 20 joueurs sur la période 2003-2020",
        line=dict(color='steelblue', dash='dash')
    ))
    
    # Taux de réussite pour le joueur spécifié (LeBron James par défaut) 
    fig.add_trace(go.Scatter(
        x=ct.index,
        y=ct['successRate_player'],
        mode='lines+markers',
        name=f"{player}",
        line=dict(color='lime')
    ))

     # Taux de réussite global
    fig.add_trace(go.Scatter(
        x=ct.index,
        y=[mean_percent] * len(ct.index),
        mode='lines',
        name="Taux de réussite global",
        line=dict(color='black', dash='solid', width=.5)
    ))

    fig.update_layout(
        title=f"Tx de réussite selon l'action précédente",
        title_x=0.5,  
        title_y=1, 
        title_xanchor='center', 
        xaxis_title="Actions qui précèdent les tirs",
        yaxis_title="Taux de réussite des tirs",
        xaxis=dict(
            showgrid=False,  
            zeroline=False, 
            showline=False,
            tickangle=45  
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        template='plotly_dark',  
        width=1000,
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            y=1.05,
            x=.5,
            xanchor='center',
            yanchor='bottom'
        )  
    )

    return fig

@st.cache_data
def accuracy_by_zone(df, player_name, label):
    df = df.loc[df.player.str.upper() == player_name.upper()] 
    df['Xloc'] =  df.apply(lambda x: round(x['Xloc']/10), axis=1)
    df['Yloc'] =  df.apply(lambda x: round(x['Yloc']/10), axis=1)

    df = df.groupby(['Xloc', 'Yloc']).agg({'target':['mean', 'count']}).\
        droplevel(0, axis=1).reset_index()

    df[label] = df.apply(lambda x: round(x['mean'], 2), axis=1)

    # Taille des points (formule à ajuster en fonction du rendu souhaité)
    df['scaled'] = (df['count'] ** 1.2).apply(math.sqrt)
    
    fig = px.scatter(df, x='Xloc', y='Yloc',
                    color=label, size='scaled',
                    color_continuous_scale='viridis',
                    hover_data={'count': True, 'scaled': False})
    
    fig.update_xaxes(range=[-27, 27], autorange=False)
    fig.update_yaxes(range=[-5, 49], autorange=False)
    fig.update_layout(title="Précision au tir par zone",
                      showlegend=True, 
                      width=500, height=500)
    fig.update_traces(marker=dict(line={'width':0.2}))

    return fig

@st.cache_data
def plotly_shots_by_year(df, player_name="Lebron James", year=2003):
    # Filtrer les données du joueur et de l'année
    player_stats = df.loc[(df.player.str.upper() == player_name.upper()) & (df.year == year)]

    # Définir les couleurs pour les tirs réussis et manqués
    colors = {1: 'lime', 0: 'lightsteelblue'}
    points_colors = [colors[i] for i in player_stats.target]
    
    # Créer la figure Plotly
    fig = go.Figure()
    
    # Ajouter les tirs du joueur
    fig.add_trace(go.Scatter(
        x=player_stats.Xloc, 
        y=player_stats.Yloc,
        mode='markers',
        marker=dict(
            color=points_colors,
            size=9,
            symbol='diamond',
            opacity=0.3
        ),
    ))
    draw_plotly_court(fig, fig_width=500, margins=30)

    fig.update_layout(
        title=f'Les tirs de {player_name} en {year}',
        title_x=0.5,  
        title_y=0.90,
        title_xanchor='center', 
        title_yanchor='top', 
        template="plotly_dark",
        showlegend=False,
        width=500,
        height=600,
        xaxis=dict(
            range=[-250, 250], 
            showticklabels=False, 
            zeroline=False
        ),
        yaxis=dict(
            range=[-50, 425], 
            showticklabels=False, 
            zeroline=False
        ),
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def plotly_shots_global(df, player_name="Lebron James"):

    player_stats = df.loc[(df.player.str.upper() == player_name.upper())]

    colors = {1: 'lime', 0: 'lightsteelblue'}
    points_colors = [colors[i] for i in player_stats.target]
        
    fig = go.Figure()    
    
    fig.add_trace(go.Scatter(
        x=player_stats.Xloc, 
        y=player_stats.Yloc,
        mode='markers',
        marker=dict(
            color=points_colors,
            size=9,
            symbol='diamond',
            opacity=0.2
        ),
    ))
    draw_plotly_court(fig, fig_width=500, margins=30)
    fig.update_layout(
        title=f'Les tirs de {player_name} sur la carrière',
        title_x=0.5,  
        title_y=1,
        title_xanchor='center', 
        title_yanchor='top', 
        template="plotly_dark",
        showlegend=False,
        width=500,
        height=500,
        xaxis=dict(
            range=[-250, 250], 
            showticklabels=False, 
            zeroline=False
        ),
        yaxis=dict(
            range=[-50, 425], 
            showticklabels=False, 
            zeroline=True
        ),
        plot_bgcolor="rgba(0,0,0,0)"
    )

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
def players_shots_by_year(df, player_name="Lebron James", year=2003):
    player_stats = df.loc[(df.player.str.upper() == player_name.upper()) & (df.year == year)]    
    colors = {1: 'lime', 0: 'lightsteelblue'}
    points_colors = [colors[i] for i in player_stats.target]
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_facecolor((1, 0.7, 0.1, 0.8))
    ax.set_xlim(-250, 250)
    ax.set_ylim(422.5, -47.5)
    
    ax.set_xlim((-250, 250)[::-1])
    ax.set_ylim((422.5, -47.5)[::-1])
    
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_title(f'Les tirs de {player_name} en {year}', fontsize=5)
        
    draw_court(ax, color='black', lw=.7, shotzone=False, outer_lines=False, alpha = 0.9)
        
    x_made = player_stats.Xloc
    y_made = player_stats.Yloc
    ax.scatter(x_made, y_made,
               c = points_colors,
               marker = 'p',
               s=20,
               linewidths=3,
               alpha = 0.2)
    
    return fig


def draw_court(_ax=None, color="blue", lw=1, shotzone=False, outer_lines=False, alpha = 1, alpha2 = 1):
    if _ax is None:
        _ax = plt.gca()

    # Create the various parts of an NBA basketball court
    _ax.set_facecolor('#FFFFFF') 
    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, linestyle='-', color=color, fill=False, alpha = alpha)

    # Create backboard
    backboard = Rectangle((-30, -12.5), 60, 0, linewidth=lw, linestyle='-', color=color, alpha = alpha)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False, alpha = alpha)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False, alpha = alpha)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False, alpha = alpha)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed', alpha = alpha)
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color, alpha = alpha)

    # Three point line
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color, alpha = alpha)
    # Create the right side 3pt lines, it's 14ft long before it arcs
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color, alpha = alpha)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color, alpha = alpha)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color, alpha = alpha)
    
    # Draw shotzone Lines
    # Based on Advanced Zone Mode
    if (shotzone == True):
        inner_circle = Circle((0, 0), radius=80, linewidth=lw, color='gray', fill=False, alpha = alpha2)
        outer_circle = Circle((0, 0), radius=160, linewidth=lw, color='gray', fill=False, alpha = alpha2)
        corner_three_a_x =  Rectangle((-250, 92.5), 30, 0, linewidth=lw, color='gray', alpha = alpha2)
        corner_three_b_x = Rectangle((220, 92.5), 30, 0, linewidth=lw, color='gray', alpha = alpha2)
        
        # 60 degrees
        inner_line_1 = Rectangle((40, 69.28), 80, 0, 60, linewidth=lw, color='gray', alpha = alpha2)
        # 120 degrees
        inner_line_2 = Rectangle((-40, 69.28), 80, 0, 120, linewidth=lw, color='gray', alpha = alpha2)
        
        # Assume x distance is also 40 for the endpoint
        inner_line_3 = Rectangle((53.20, 150.89), 290, 0, 70.53, linewidth=lw, color='gray', alpha = alpha2)
        inner_line_4 = Rectangle((-53.20, 150.89), 290, 0, 109.47, linewidth=lw, color='gray', alpha = alpha2)
        
        # Assume y distance is also 92.5 for the endpoint
        inner_line_5 = Rectangle((130.54, 92.5), 80, 0, 35.32, linewidth=lw, color='gray', alpha = alpha2)
        inner_line_6 = Rectangle((-130.54, 92.5), 80, 0, 144.68, linewidth=lw, color='gray', alpha = alpha2)
        
        
        # List of the court elements to be plotted onto the axes
        court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                          bottom_free_throw, restricted, corner_three_a,
                          corner_three_b, three_arc, center_outer_arc, inner_circle, outer_circle,
                          corner_three_a_x, corner_three_b_x,
                          inner_line_1, inner_line_2, inner_line_3, inner_line_4, inner_line_5, inner_line_6]
    else:
        # List of the court elements to be plotted onto the axes
        court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                          bottom_free_throw, restricted, corner_three_a,
                          corner_three_b, three_arc, center_outer_arc,]
    

    # Add the court elements onto the axes
    for element in court_elements:
        _ax.add_patch(element)
        

    return _ax

@st.cache_data
def players_shots_global(df, player_name="Lebron James"):
    player_stats = df.loc[df.player.str.upper() == player_name.upper()]    
    colors = {1: 'lime', 0: 'lightsteelblue'}
    points_colors = [colors[i] for i in player_stats.target]
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_facecolor((1, 0.7, 0.1, 0.8))
    ax.set_xlim(-250, 250)
    ax.set_ylim(422.5, -47.5)
    
    ax.set_xlim((-250, 250)[::-1])
    ax.set_ylim((422.5, -47.5)[::-1])
    
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_title(f'Les tirs de {player_name} sur l\'ensemble de la période', fontsize=12)
        
    draw_court(ax, color='black', lw=.7, shotzone=False, outer_lines=False, alpha = 0.9)
        
    x_made = player_stats.Xloc
    y_made = player_stats.Yloc
    ax.scatter(x_made, y_made,
               c = points_colors,
               marker = 'p',
               s=20,
               linewidths=3,
               alpha = 0.2)
    
    return fig