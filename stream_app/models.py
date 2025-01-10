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
    df = pd.read_csv('../_PRE_PRO_DATAS/df_one_hot_encoded.csv', index_col=0)
    # Encodage avec StandatdScaler et séparation des données
    X, y, X_train, X_test, y_train, y_test, feature_names = setting_scaling_dataframes(df)
    rows, cols = df.shape
    with st.expander("Voir un extrait des données d'entraînement encodées"):
        st.write(f"Le dataframe contient {rows} lignes et {cols} colonnes.")
        st.dataframe(X_train.head())
    
    
    st.subheader('II. Machine Learning')
    # Tableau qui contient les scores des classifieurs
    scores = pd.DataFrame(columns=["Classifieur", "Dataset", "Accuracy", \
                                "F1-score (classe négative)", "F1-score (classe positive)"])
    
    # Affichage des scores des classifieurs
    def display_scores(model, accuracy, f1_0, f1_1):
        st.markdown(f"""
        <h4 style='text-align: center; background-color: #1D428A; color: white;'>{model}</h4>
        <p style='text-align: center;'>Précision du modèle</p>
        <p style='text-align: center; color:#4CAF50;'>{accuracy}</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 0</p>
        <p style='text-align: center; color:#4CAF50;'>{f1_0}</p>
        <hr style="border: 1px solid #cccccc; margin: 20px 0;"> 
        <p style='text-align: center;'>f1 score sur la classe 1</p>
        <p style='text-align: center; color:#4CAF50;'>{f1_1}</p>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5, border=True)

    with col1:
        # Régression logistique
        model = """
        Régression 
        
        Logistique"""

        model_path = 'models_saved/lr_model.joblib'

        if os.path.exists(model_path):
            lr_model = load(model_path)
            lr_preds = lr_model.predict(X_test)
        else:
            print(f"Le modèle n'existe pas. Entraînement d'un nouveau modèle...")
            lr_model, lr_preds = lr_classification(X_train, X_test, y_train, C=.1, solver='liblinear', save=True)

        report, accuracy, f1_0, f1_1 = classification_scores(y_test, lr_preds)
        scores.loc[0] = ["Régression logistique", "Dataset complet", accuracy, f1_0, f1_1]
        display_scores(model, accuracy, f1_0, f1_1)

    with col2:
        model = """
        Arbre 
        
        de décision"""

        model_path = 'models_saved/dt_model.joblib'

        if os.path.exists(model_path):
            dt_model = load(model_path)
            dt_preds = dt_model.predict(X_test)
        else:
            print(f"Le modèle n'existe pas. Entraînement d'un nouveau modèle...")
            dt_model, dt_preds = dt_classification(X_train, X_test, y_train, max_depth=11, random_state=123, save=True)
        
        report, accuracy, f1_0, f1_1 = classification_scores(y_test, dt_preds)
        scores.loc[1] = ["Arbre de décision", "Dataset complet", accuracy, f1_0, f1_1]
        display_scores(model, accuracy, f1_0, f1_1)

    with col3: 
        model = """
        Random

        Forest"""
        
        model_path = 'models_saved/rf_model.joblib'

        if os.path.exists(model_path):
            rf_model = load(model_path)
            rf_preds = rf_model.predict(X_test)
        else:
            print(f"Le modèle n'existe pas. Entraînement d'un nouveau modèle...")
            rf_model, rf_preds = rf_classification(X_train, X_test, y_train, n_jobs=-1, n_estimators=300, max_depth=20, random_state=123, save=True)

        report, accuracy, f1_0, f1_1 = classification_scores(y_test, rf_preds)
        scores.loc[2] = ["Random Forest", "Dataset complet", accuracy, f1_0, f1_1]
        display_scores(model, accuracy, f1_0, f1_1)

    with col4:
        model = """
        eXtreme 
        
        Gradient Boost"""
        
        model_path = 'models_saved/xgb_model.joblib'

        if os.path.exists(model_path):
            # Charger le modèle
            xgb_model = load(model_path)
            xgb_preds = xgb_model.predict(X_test)
        else:
            print(f"Le modèle n'existe pas. Entraînement d'un nouveau modèle...")
            xgb_model, xgb_preds = xgb_classification(X_train, X_test, y_train, max_depth=9, eta=.1, min_child_weight=4, gamma=.05, save=True)

        report, accuracy, f1_0, f1_1 = classification_scores(y_test, xgb_preds)
        scores.loc[3] = ["XGBoost", "Dataset complet", accuracy, f1_0, f1_1]
        display_scores(model, accuracy, f1_0, f1_1)

    with col5:
        model = """
        Deep 
        
        Neural Network"""

        model_path = 'models_saved/dnn_model.h5'

        if os.path.exists(model_path):
            # Charger le modèle
            dnn_model = load_model(model_path, compile=False)
            dnn_probs = dnn_model.predict(X_test)
            dnn_preds = (dnn_probs > .473).astype(int).flatten()

        else:
            print(f"Le modèle n'existe pas. Entraînement d'un nouveau modèle...")
            dnn_model, dnn_preds = dnn_classification(X_train, X_test, y_train, save=True)

        report, accuracy, f1_0, f1_1 = classification_scores(y_test, dnn_preds)
        scores.loc[4] = ["DNN", "Dataset complet", accuracy, f1_0, f1_1]

        display_scores(model, accuracy, f1_0, f1_1)


    st.subheader("III. Interprétation du modèle eXtreme Gradient Boost avec SHAP")
 
   
    _explanation, explanation_obj, shap_values, base_value = SHAP_explanations(_model = xgb_model, 
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

        fig = SHAP_summary_plot(_explanation, shap_values, X_test, feature_names, max_display=nb_features)
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


    