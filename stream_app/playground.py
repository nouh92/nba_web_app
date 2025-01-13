#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
                      ### PAGE Playground ###
#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import StandardScaler
from src.classification import *

# Fichier de données
data_file = '_PRE_PRO_DATAS/df_quarter_size.csv'

# Liste des modèles
models = ['Régression logistique', 'Arbre de décision', 'Random Forest',\
          'eXtreme Gradient Boosting', 'Deep Neural Network']

# Paramètres des modèles
lr_C = [.1, .5, 1, 10, 100]
lr_solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
dt_max_depth = range(5, 16)
rf_estimators = [50, 100, 200, 300, 400, 500, 600]
rf_max_depth = [10, 15, 20, 25, 30]
xgb_eta = [.01, .05, .1, .2, .5, 1, 5]
xgb_max_depth= range(8, 13)
xgb_min_child_weight = range(1, 10)
xgb_gamma = [0, .01, .05, .1, 1]
random_state = [42, 123, 789]

# Dictionnaire des joueurs
players = {2544 : 'LeBron James',
           201142 : 'Kevin Durant',
           201935 : 'James Harden',
           201939 : 'Stephen Curry',
           201942 : 'DeMar DeRozan',
           201950 : 'Jrue Holiday', 
           202331 : 'Paul George',
           202681 : 'Kyrie Irving',
           202685 : 'Jonas Valanciunas',
           202691 : 'Klay Thompson',
           202695 : 'Kawhi Leonard',
           202710 : 'Jimmy Butler',
           203076 : 'Anthony Davis',
           203078 : 'Bradley Beal',
           203081 : 'Damian Lillard',
           203110 : 'Draymond Green',
           203114 : 'Khris Middleton',
           203484 : 'Kentavious Caldwell-Pope',
           203497 : 'Rudy Gobert',
           203507 : 'Giannis Antetokounmpo'}


#####################################
# Lecture du fichier CSV
#####################################

def read_file(data_file):
  return pd.read_csv(data_file, index_col=0)


#####################################
# Préparation de la classification
#####################################

def prepare_classification(df):

  X, y = df.drop(['target'], axis=1), df.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

  cols_to_scale = ['Xloc', 'Yloc', 'shotDistance', 'shotPoints']
  scaler = StandardScaler()
  X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
  X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_scaled[cols_to_scale])
  X_test_scaled[cols_to_scale] = scaler.transform(X_test_scaled[cols_to_scale])

  return X_train_scaled, X_test, X_test_scaled, y_train, y_test


#####################################
# Lancement de la classification
#####################################

def launch_classification(df, selected_player, selected_model,
                          param1=None, param2=None, param3=None, param4=None):

  # Sélection des données
  # On écarte les tirs de l'autre côté du terrain
  df = df[(df['player_' + selected_player.upper()] == 1) & (df.Yloc <= 417.5)]

  # Préparation de la classification
  X_train_scaled, X_test, X_test_scaled, y_train, y_test = prepare_classification(df)

  # Classification
  if selected_model == models[0]:
    _, y_preds = lr_classification(X_train_scaled, X_test_scaled, y_train, param1, param2)
  if selected_model == models[1]:
    _, y_preds = dt_classification(X_train_scaled, X_test_scaled, y_train, param1, param2)
  if selected_model == models[2]:
    _, y_preds = rf_classification(X_train_scaled, X_test_scaled, y_train, param1, param2, param3, param4)
  if selected_model == models[3]:
    _, y_preds = xgb_classification(X_train_scaled, X_test_scaled, y_train, param1, param2, param3, param4)
  if selected_model == models[4]:
    _, y_preds = dnn_classification(X_train_scaled, X_test_scaled, y_train)
  
  # Récupération des résultats
  df_preds = X_test.copy()
  df_preds['result'], df_preds['prediction'] = y_test, y_preds
  df_preds['target'] = df_preds.apply(lambda x: 1 if x.result == x.prediction else 0, axis=1)
  _, accuracy, f1_0, f1_1 = classification_scores(y_test, y_preds)

  return df_preds, accuracy, f1_0, f1_1


#####################################
# Couleurs des scores
#####################################

def set_colors(m1_accuracy, m1_f1_0, m1_f1_1, m2_accuracy=0, m2_f1_0=0, m2_f1_1=0):
    
    # Accuracy
    if m2_accuracy == m1_accuracy: m1_accuracy_color, m2_accuracy_color = "blue", "blue"
    elif m1_accuracy > m2_accuracy: m1_accuracy_color, m2_accuracy_color = "green", "red"
    else: m1_accuracy_color, m2_accuracy_color = "red", "green"
    
    # F1-score classe 0
    if m2_f1_0 == m1_f1_0: m1_f1_0_color, m2_f1_0_color = "blue", "blue"
    elif m1_f1_0 > m2_f1_0: m1_f1_0_color, m2_f1_0_color = "green", "red"
    else: m1_f1_0_color, m2_f1_0_color = "red", "green"
    
    # F1-score classe 1
    if m2_f1_1 == m1_f1_1: m1_f1_1_color, m2_f1_1_color = "blue", "blue"
    elif m1_f1_1 > m2_f1_1: m1_f1_1_color, m2_f1_1_color = "green", "red"
    else: m1_f1_1_color, m2_f1_1_color = "red", "green"
        
    return(m1_accuracy_color, m1_f1_0_color, m1_f1_1_color,
           m2_accuracy_color, m2_f1_0_color, m2_f1_1_color)


#####################################
# Arrondis pour Xloc et Yloc
#####################################

def round_loc(row):
  # Les coordonnées X et Y sont exprimées en dixièmes de pieds, soit environ 3 centimètres
  # En arrondissant avec la variable "step", on réduit la granularité
  # Par exemple, si step = 10, l'unité devient 10 * 0.1 pied = 1 pied = 30 centimètres
  step = 10 if row['shotZoneBasic_Restricted Area'] == 1 else 15
  row['Xloc'], row['Yloc'] = row.Xloc//step*step, row.Yloc//step*step
  return row


#####################################
# Construction du graphique
#####################################

def build_fig(df, label='target'):

  # On arrondit Xloc et Yloc
  df = df.apply(round_loc, axis = 1)

  # Coefficient de réduction pour les joueurs les plus anciens
  coeff = .7 if df.year_2007.max() == 1 else .8

  # Moyenne et somme de la variable cible pour chaque coordonnée
  df = df.groupby(['Xloc', 'Yloc']).agg({'target':['mean', 'count']}).\
    droplevel(0, axis=1).reset_index()

  # Renommage et arrondi de la moyenne
  df = df.rename(columns={'mean' : label})
  df[label] = df.apply(lambda x: round(x[label], 2), axis=1)

  # Taille des points (règle de calcul à ajuster en fonction du rendu souhaité)
  df['scaled'] = df.apply(lambda x: 5*x['count']**coeff, axis=1)

  # Affichage des données
  fig = px.scatter(df, x='Xloc', y='Yloc',
                   color=label, color_continuous_scale='bluered',
                   hover_data={'count': True, 'scaled': False})
  
  # Affichage du terrain de basket
  draw_plotly_court(fig)

  # Mise en forme
  fig.update_xaxes(title_text="")
  fig.update_yaxes(title_text="")
  fig.update_traces(marker=dict(line={'width':0.2}, size=df['scaled']))

  return fig


##########################################################################
# Affichage du terrain de basket avec Plotly
# https://gist.github.com/jpolarizing/17a8ceb6d49d45140ebbcea6f59c73ac
##########################################################################

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
      paper_bgcolor="white", plot_bgcolor="white",
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


#####################################
# Fonction principale
#####################################

def main():
  st.cache_data.clear()

  # Modifications de styles
  css="""
    <style> 
    .block-container {padding-top: 0px; padding-bottom: 0px; padding-left: 20px; padding-right: 0px}
    .st-ct > div {background-color: #fff}
    .st-dg > div {background-color: #fff}
    div[data-testid="stMarkdownContainer"] {font-size: 16px}
    </style>
  """
  st.write(css, unsafe_allow_html=True)

  # En-tête
  st.title("Evaluation des modèles")
  st.subheader("Testez et comparez les modèles pour chaque joueur")

  # Paramétrage du premier modèle
  m1_c1, m1_c2, m1_c3, m1_c4, m1_c5, _ = st.columns([1.6,.8,.8,.8,.8,.8], gap='small')
  m1_p1, m1_p2, m1_p3, m1_p4 = None, None, None, None
  m1 = m1_c1.selectbox("Modèle 1", options=models,
                       index=None, placeholder='Veuillez choisir un premier modèle')
  if m1 == models[0]:
      m1_p1 = m1_c2.selectbox('C', options=lr_C, index=0, key='lr1_0')
      m1_p2 = m1_c3.selectbox('solver', options=lr_solver, index=1, key='lr1_1')
  if m1 == models[1]:
      m1_p1 = m1_c2.selectbox('max_depth', options=dt_max_depth, index=6, key='dt1_0')
      m1_p2 = m1_c3.selectbox('random_state', options=random_state, index=1, key='dt1_1')
  if m1 == models[2]:
      m1_p2 = m1_c2.selectbox('n_estimators', options=rf_estimators, index=3, key='rf1_0')
      m1_p3 = m1_c3.selectbox('max_depth', options=rf_max_depth, index=2, key='rf1_1')
      m1_p4 = m1_c4.selectbox('random_state', options=random_state, index=1, key='rf1_2')
  if m1 == models[3]:
      m1_p1 = m1_c2.selectbox('max_depth', options=xgb_max_depth, index=1, key='xgb1_0')
      m1_p2 = m1_c3.selectbox('eta', options=xgb_eta, index=2, key='xgb1_1')
      m1_p3 = m1_c4.selectbox('min_child_weight', index=3, options=xgb_min_child_weight, key='xgb1_2')
      m1_p4 = m1_c5.selectbox('gamma', options=xgb_gamma, index=2, key='xgb1_3')

  # Paramétrage du second modèle (optionnel)
  m2_c1, m2_c2, m2_c3, m2_c4, m2_c5, _ = st.columns([1.6,.8,.8,.8,.8,.8], gap='small')
  m2_p1, m2_p2, m2_p3, m2_p4 = None, None, None, None
  m2 = m2_c1.selectbox("Modèle 2 (optionnel)", options=models,
                       index=None, placeholder='Choisir un second modèle (optionnel)')
  if m2 == models[0]:
      m2_p1 = m2_c2.selectbox('C', options=lr_C, index=0, key='lr2_0')
      m2_p2 = m2_c3.selectbox('solver', options=lr_solver, index=1, key='lr2_1')
  if m2 == models[1]:
      m2_p1 = m2_c2.selectbox('max_depth', options=dt_max_depth, index=6, key='dt2_0')
      m2_p2 = m2_c3.selectbox('random_state', options=random_state, index=1, key='dt2_1')
  if m2 == models[2]:
      m2_p2 = m2_c2.selectbox('n_estimators', options=rf_estimators, index=3, key='rf2_0')
      m2_p3 = m2_c3.selectbox('max_depth', options=rf_max_depth, index=2, key='rf2_1')
      m2_p4 = m2_c4.selectbox('random_state', options=random_state, index=1, key='rf2_2')
  if m2 == models[3]:
      m2_p1 = m2_c2.selectbox('max_depth', options=xgb_max_depth, index=1, key='xgb2_0')
      m2_p2 = m2_c3.selectbox('eta', options=xgb_eta, index=2, key='xgb2_1')
      m2_p3 = m2_c4.selectbox('min_child_weight', index=3, options=xgb_min_child_weight, key='xgb2_2')
      m2_p4 = m2_c5.selectbox('gamma', options=xgb_gamma, index=2, key='xgb2_3')

  # Choix du joueur
  col1, col2, _, _, _, _ = st.columns([1.6,1.6,.6,.6,.6,.6],
                                      gap='small', vertical_alignment='bottom')
  player = col1.selectbox("Joueur", options=sorted(pd.Series(players)),
                            index=None, placeholder='Veuillez choisir un joueur')
  
  # Bouton pour lancer la modélisation
  disabled = True if ((player == None) | (m1 == None)) else False
  result = col2.button("Lancer la modélisation", disabled=disabled)
  st.write("")

  # Lancement des classifications
  if result & (player != None) & (m1 != None) :
    df = read_file(data_file)
    m1_preds, m1_accuracy, m1_f1_0, m1_f1_1 = \
      launch_classification(df, player, m1, m1_p1, m1_p2, m1_p3, m1_p4)
    m2_accuracy, m2_f1_0, m2_f1_1 = m1_accuracy, m1_f1_0, m1_f1_1
    if m2!=None:
           m2_preds, m2_accuracy, m2_f1_0, m2_f1_1 = \
            launch_classification(df, player, m2, m2_p1, m2_p2, m2_p3, m2_p4)
    m1_accuracy_color, m1_f1_0_color, m1_f1_1_color, \
      m2_accuracy_color, m2_f1_0_color, m2_f1_1_color = \
        set_colors(m1_accuracy, m1_f1_0, m1_f1_1, m2_accuracy, m2_f1_0, m2_f1_1)      

  # Résultats du premier modèle
  if result & (player != None) & (m1 != None) :
    left_col, right_col = st.columns(2)
    with left_col:
      st.markdown(f'''**Modèle 1 : {m1}**  
                  Précision du modèle : <span style="color: {m1_accuracy_color}">{m1_accuracy}</span>  
                  F1-score sur la classe 0 : <span style="color: {m1_f1_0_color}">{m1_f1_0}</span>  
                  F1-score sur la classe 1 : <span style="color: {m1_f1_1_color}">{m1_f1_1}</span>  
                  Accuracy en fonction de la position sur le terrain :''', unsafe_allow_html=True)
      st.plotly_chart(build_fig(m1_preds, label='accuracy'), key='fig1')
      st.write("Taux de prédictions correctes pour les tirs manqués :")
      st.plotly_chart(build_fig(m1_preds[m1_preds.result == 0], label='accuracy'), key='fig2')
      st.write("Taux de prédictions correctes pour les tirs réussis :")
      st.plotly_chart(build_fig(m1_preds[m1_preds.result == 1], label='accuracy'), key='fig3')

  # Résultats du second modèle
  if result & (player != None) & (m1 != None) & (m2 != None) :
    with right_col:
      st.markdown(f'''**Modèle 2 : {m2}**  
                  Précision du modèle : <span style="color: {m2_accuracy_color}">{m2_accuracy}</span>  
                  F1-score sur la classe 0 : <span style="color: {m2_f1_0_color}">{m2_f1_0}</span>  
                  F1-score sur la classe 1 : <span style="color: {m2_f1_1_color}">{m2_f1_1}</span>   
                  Accuracy en fonction de la position sur le terrain :''', unsafe_allow_html=True)
      st.plotly_chart(build_fig(m2_preds, label='accuracy'), key='fig4')
      st.write("Taux de prédictions correctes pour les tirs manqués :")
      st.plotly_chart(build_fig(m2_preds[m2_preds.result == 0], label='accuracy'), key='fig5')
      st.write("Taux de prédictions correctes pour les tirs réussis :")
      st.plotly_chart(build_fig(m2_preds[m2_preds.result == 1], label='accuracy'), key='fig6')

  st.cache_data.clear()
