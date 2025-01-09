import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import *
from src.classification import *
from src.dashboard_functions import *



# Pour afficher les stats de la carrière
def whole_stats(df, player_data, player_name):
    if st.button("Voir les datas pour la saison", type='primary', icon="🔥", use_container_width=True):
        st.session_state.view = "saison"
        st.rerun()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.subheader(f'Les stats en carrière')
    with col3:
        st.write(' ')

    col2a, col2b = st.columns(2, vertical_alignment="center")
         
    with col2a:
      shot_success_rate = player_data.target.mean() * 100
      st.markdown(f"""
          <p style="font-size: 16px; font-weight: bold;">
            <span>
                  Taux de réussite au tir en carrière :
            </span> 
            <span style="color: green; font-size: 32px; padding: 10px">{shot_success_rate:.2f}%</span>
          </p>
      """, unsafe_allow_html=True)

    with col2b:
      fig = plot_target_repartition(player_data)
      st.plotly_chart(fig)

    col2a, col2b = st.columns(2)
    with col2a:
      fig = plotly_shots_global(df, player_name=player_name)
      st.plotly_chart(fig)
    
    with col2b:
      shot_labels = pd.DataFrame(player_data.shotLabel.value_counts())
      fig = histo_horiz(shot_labels)
      st.plotly_chart(fig)

    col2a, col2b = st.columns(2, vertical_alignment="top")
    with col2a:
      fig = player_shots_stats_per_game(df=player_data, player_name=player_name)
      st.plotly_chart(fig)

    
    with col2b:
      fig = previous_event_viz(df, player=player_name)
      st.plotly_chart(fig)



# Pour afficher les stats de la saison

def yearly_stats(df, player_name, available_years):
    if st.button("Voir les datas pour toute la carrière", type='primary', icon="🔥", use_container_width=True):
        st.session_state.view = "carrière"
        st.rerun()

    selected_year = st.slider("", 
        min_value=available_years[0], 
        max_value=available_years[-1], 
        value=st.session_state.selected_year, 
        step=1
        )

    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.subheader(f'Les stats pour la saison {st.session_state.selected_year}')
    with col3:
        st.write(' ')
    
    player_data_year = df[(df.player == player_name) & (df.year == st.session_state.selected_year)]

    col2a, col2b = st.columns(2, vertical_alignment="center")

    with col2a:
        shot_success_rate_year = player_data_year.target.mean() * 100
        st.markdown(f"""
          <p style="font-size: 16px; font-weight: bold;">
              <span>
                    Taux de réussite au tir sur la saison :
              </span> 
              <span style="color: green; font-size: 32px; padding: 10px">{shot_success_rate_year:.2f}%</span>
          </p>
        """, unsafe_allow_html=True)

    with col2b:
        fig = plot_target_repartition(player_data_year)
        st.plotly_chart(fig)

    col2a, col2b = st.columns(2, vertical_alignment="center")
    with col2a:
          fig = plotly_shots_by_year(df, player_name=player_name, year=st.session_state.selected_year)
          st.plotly_chart(fig)
    
    with col2b:
        shot_labels_year = pd.DataFrame(player_data_year.shotLabel.value_counts())
        fig = histo_horiz(shot_labels_year)
        st.plotly_chart(fig)


# Fonction principale
def main():

  df = set_dataframe()

  st.title('Bienvenue sur le tableau de stats par joueur')
  col1, col2 = st.columns([.8,3.2], gap='large')

  if 'selected_player' not in st.session_state:
    st.session_state.selected_player = 'LEBRON JAMES'



  with col1:
    st.subheader("Joueur")

    def selections(player_name):
        st.session_state.selected_player = player_name
        st.session_state.selected_year = None 
        st.rerun() 

    for player in df.player.unique():
        if st.button(player, type='tertiary'):
          selections(player)

  with col2:
    if 'selected_player' in st.session_state:
        selected_player = st.session_state.selected_player
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.header(selected_player)
        with col3:
            st.write(' ')
        
        player_data = df[df['player'] == selected_player]
        available_years = sorted(player_data.year.unique())

    if 'selected_year' not in st.session_state or st.session_state.selected_year not in available_years:
        st.session_state.selected_year = available_years[0]
  
  
    player_name = selected_player.upper()
    player_data = df[(df.player == player_name)]

    
    # Initialiser l'état de la vue s'il n'est pas déjà dans la session
    if "view" not in st.session_state:
        st.session_state.view = "saison"  # Valeur initiale de la vue, "saison" par défaut

    # Afficher la vue selon l'état actuel dans session_state
    if st.session_state.view == "saison":
        yearly_stats(df, player_name, available_years)
    elif st.session_state.view == "carrière":
        whole_stats(df, player_data, player_name)







  
