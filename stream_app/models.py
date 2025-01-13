#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
                      ### PAGE Modélisation ###
#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

import streamlit as st

import os
from joblib import load

import tensorflow as tf
from tensorflow.keras.models import Model, load_model



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import *
from src.classification import *
from src.merge_and_encode import *

print(f"Current Working Directory: {os.getcwd()}")

def apply_custom_styles():
    st.markdown("""
    <style>
    /* Modifier le fond et la couleur du texte pour le champ number_input */
    .stNumberInput input {
        background-color: white;
        color: black;
        border: 1px solid #ccc;
        padding: 5px;
        font-size: 16px;
    }

    /* Modifier les boutons + et - */
    .stNumberInput button {
        background-color: white;
        color: black;
        border: 1px solid #ccc;
        padding: 5px;
        font-size: 16px;
        cursor: pointer;
    }

    /* Changer le fond et la couleur des boutons au survol */
    .stNumberInput button:hover {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    apply_custom_styles()
    st.title('Modélisation')
    st.write(
        """
        Cinq modèles de classification sont entraînés puis utilisés afin de déterminer quel est le plus efficace pour prédire le résultat des tirs sur notre dataframe :
        - le modèle de régression logistique de Scikit-learn
        - le modèle Random Forest de Scikit-learn
        - le modèle open-source eXtreme Gradient Boosting (XGBoost) développép Tianqi Chen, chercheur en machine learning et en intelligence artificielle de l'université de Washington
        - le modèle Deep Neural Network de Keras
        - l'arbre de décision de Scikit-learn
        """
    )

    st.subheader('I. Préparation des données : encoding et séparation')
    st.write(
        """
        Avant d'utiliser les différents classifieurs, le nouveau dataframe fait l'objet d'une nouvelle brève analyse exploratoire ; puis les données sont encodées et séparées en un jeu de d'entrainement et un jeu de test. Ce dernier représente 20% de l'ensemble des données.
        """
    )

    df = pd.read_csv('_PRE_PRO_DATAS/merged.csv', index_col=0).sort_index()

    with st.expander("Nouveaux tests Khi2 sur les variables catégorielles"):
        var_list = ['shotZoneBasic', 'shotZoneArea', 'shotZoneRange', 'player', 'previousEvent', 'shotType']
        target = 'target'
        chi2_testing(df=df, var_list=var_list, target=target)
    
    with st.expander("Heatmap des variables numériques pour déterminer les corrélations"):
        heatmap(df)
    
    st.markdown('**Encoding**')

    st.write(
        """
        Les variables non numériques sont encodées avec OneHotEncoder de Scikit-learn : shotType, shotZoneBasic, shotZoneArea, shotZoneRange, player. 

        Le StandardScaler de Scikit-learn  est utilisé pour les variables numériques suivantes : Xloc, Yloc, shotDistance et shotPoints.
        """
    )

    
    # Chargement du datasets encodé avec OneHotEncoder
    # df = pd.read_csv('_PRE_PRO_DATAS/df_one_hot_encoded.csv', index_col=0)
    # Encodage avec StandatdScaler et séparation des données
    # X, y, X_train, X_test, y_train, y_test, feature_names = setting_scaling_dataframes(df)
    X_train = pd.read_csv('_PRE_PRO_DATAS/X_train.csv', index_col=0)
    rows, cols = df.shape
    with st.expander("Voir un extrait des données d'entraînement encodées"):
        st.write(f"Le dataframe contient {rows} lignes et {cols} colonnes.")
        st.dataframe(X_train.head())
    
    
    st.subheader('II. Machine Learning')
    
    
    col1, col2, col3, col4, col5 = st.columns(5, border=True)

    with col1:
        # Régression logistique
        model = """
        Régression 
        
        Logistique"""

        st.markdown(f"""
        <h4 style='text-align: center; background-color: #1D428A; color: white;'>{model}</h4>
        <p style='text-align: center;'>Précision du modèle</p>
        <p style='text-align: center; color:#4CAF50;'>0.647</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 0</p>
        <p style='text-align: center; color:#4CAF50;'>0.697</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 1</p>
        <p style='text-align: center; color:#4CAF50;'>0.577</p>
        """, unsafe_allow_html=True)




    with col2:
        model = """
        Arbre 
        
        de décision"""

        model_path = 'models_saved/dt_model.joblib'

        st.markdown(f"""
        <h4 style='text-align: center; background-color: #1D428A; color: white;'>{model}</h4>
        <p style='text-align: center;'>Précision du modèle</p>
        <p style='text-align: center; color:#4CAF50;'>0.645</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 0</p>
        <p style='text-align: center; color:#4CAF50;'>0.715</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 1</p>
        <p style='text-align: center; color:#4CAF50;'>0.529</p>
        """, unsafe_allow_html=True)

    with col3: 
        model = """
        Random

        Forest"""

        st.markdown(f"""
        <h4 style='text-align: center; background-color: #1D428A; color: white;'>{model}</h4>
        <p style='text-align: center;'>Précision du modèle</p>
        <p style='text-align: center; color:#4CAF50;'>0.656</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 0</p>
        <p style='text-align: center; color:#4CAF50;'>0.709</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 1</p>
        <p style='text-align: center; color:#4CAF50;'>0.579</p>
        """, unsafe_allow_html=True)

    with col4:
        model = """
        eXtreme 
        
        Gradient Boost"""
        st.markdown(f"""
        <h4 style='text-align: center; background-color: #1D428A; color: white;'>{model}</h4>
        <p style='text-align: center;'>Précision du modèle</p>
        <p style='text-align: center; color:#4CAF50;'>0.661</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 0</p>
        <p style='text-align: center; color:#4CAF50;'>0.712</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 1</p>
        <p style='text-align: center; color:#4CAF50;'>0.586</p>
        """, unsafe_allow_html=True)

    with col5:
        model = """
        Deep 
        
        Neural Network"""
        st.markdown(f"""
        <h4 style='text-align: center; background-color: #1D428A; color: white;'>{model}</h4>
        <p style='text-align: center;'>Précision du modèle</p>
        <p style='text-align: center; color:#4CAF50;'>0.658</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 0</p>
        <p style='text-align: center; color:#4CAF50;'>0.702</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 1</p>
        <p style='text-align: center; color:#4CAF50;'>0.599</p>
        """, unsafe_allow_html=True)


    st.subheader("III. Interprétation du modèle eXtreme Gradient Boost avec SHAP")

    col1, col2 = st.columns(2)

    with col1:
        st.image('img/SHAP_summary_10.png', output_format='PNG')   

    with col2:
        st.image('img/SHAP_summary_15.png', output_format='PNG')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('img/SHAP_waterfall_1.png', output_format='PNG')   

    with col2:
        st.image('img/SHAP_waterfall_5.png', output_format='PNG')
    
    with col3:
        st.image('img/SHAP_waterfall_18.png', output_format='PNG')
    

""" 
   
    _explanation, shap_values = SHAP_explanations(_model = xgb_model, 
                                                                X_train = X_train,
                                                                X_test = X_test,
                                                                preds = xgb_preds,
                                                                feature_names = feature_names,
                                                                explainer_type='TreeExplainer')
    

    col1, col2 = st.columns(2)
    with col1:
        nb_features = st.number_input(
        "Choisir un nombre de features pour le graph de Features Importances", 
        min_value=0, 
        max_value=len(feature_names)-1, 
        value=10,  # Valeur par défaut
        step=1
    )

        fig = SHAP_summary_plot(shap_values, X_test, feature_names, max_display=nb_features)
        st.pyplot(fig)
 
    with col2:
        event_index = st.number_input(
        "Entrez un index d'événement pour afficher le graphique de prise de décision du modèle :", 
        min_value=0, 
        max_value=len(_explanation)-1, 
        value=1,  # Valeur par défaut
        step=1
    )
        fig = decisionWaterfall(_explanation, event_index=event_index, max_display=10)
        st.pyplot(fig)


"""