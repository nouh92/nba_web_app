#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
                      ### PAGE Players Dashboard ###
#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import *
from src.classification import *
from src.dashboard_functions import *


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
         st.header(selected_player, divider="blue")
         player_data = df[df['player'] == selected_player]
         available_years = sorted(player_data.year.unique())

      if 'selected_year' not in st.session_state or st.session_state.selected_year not in available_years:
         st.session_state.selected_year = available_years[0]
      
      # ------------------------------------------------------------------------------------------------------------#
      # DATAS globales
      # ------------------------------------------------------------------------------------------------------------#

      player_name = selected_player.upper()
      player_data = df[(df.player == player_name)]

      with st.expander(f"Voir les datas sur toute la carrière de {player_name}", expanded=False, icon="🔥"):
         st.subheader('Data pour l\'ensemble de la période')
         
         shot_success_rate = player_data.target.mean() * 100
         st.markdown(f"""
            <p style="font-size: 16px; font-weight: bold;">
               <span>
                     Taux de réussite au tir de {player_name} sur la saison :
               </span> 
               <span style="color: green; font-size: 32px; padding: 10px">{shot_success_rate:.2f}%</span>
            </p>
         """, unsafe_allow_html=True)

         fig = plot_target_repartition(player_data)
         st.plotly_chart(fig)

         col2a, col2b = st.columns(2)
         with col2a:
            fig = player_shots_stats_per_game(player_name, player_data)
            st.plotly_chart(fig)
         
         with col2b:
            shot_labels = pd.DataFrame(player_data.shotLabel.value_counts())
            fig = histo_horiz(shot_labels)
            st.plotly_chart(fig)

         fig = previous_event_viz(df, player=player_name)
         st.plotly_chart(fig)

      # ------------------------------------------------------------------------------------------------------------#
      # DATAS annuelles
      # ------------------------------------------------------------------------------------------------------------#

      st.subheader(f'Data pour l\'année {st.session_state.selected_year}')

      selected_year = st.slider(
         "Sélection de l'année", 
         min_value=available_years[0], 
         max_value=available_years[-1], 
         value=st.session_state.selected_year, 
         step=1
         )

      if selected_year != st.session_state.selected_year:
         st.session_state.selected_year = selected_year
         st.rerun()
      
      player_data_year = df[(df.player == player_name) & (df.year == st.session_state.selected_year)]

      shot_success_rate_year = player_data_year.target.mean() * 100
      st.markdown(f"""
         <p style="font-size: 16px; font-weight: bold;">
            <span>
                  Taux de réussite au tir de {player_name} sur la saison :
            </span> 
            <span style="color: green; font-size: 32px; padding: 10px">{shot_success_rate_year:.2f}%</span>
         </p>
      """, unsafe_allow_html=True)


      fig = plot_target_repartition(player_data_year)
      st.plotly_chart(fig)


      col2a, col2b = st.columns(2)
      with col2a:
         fig = players_shots(df, player_name=selected_player, year=st.session_state.selected_year)
         if fig is not None:
               st.pyplot(fig)
      
      with col2b:
         shot_labels_year = pd.DataFrame(player_data_year.shotLabel.value_counts())
         fig = histo_horiz(shot_labels_year)
         st.plotly_chart(fig)

      
      
     




