import pandas as pd
import numpy as np

import streamlit as st

import os
from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU


from src.merge_and_encode import *

from joblib import dump


def lr_classification(X_train, X_test, y_train, C=1, solver='lbfgs', save=False):

    lr_model = LogisticRegression(C=C, solver=solver)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    if save:
        dump(lr_model, 'models_saved/lr_model.joblib')

    return lr_model, lr_preds

def dt_classification(X_train, X_test, y_train, max_depth=None, random_state=None, save=False):

    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)

    if save:
        dump(dt_model, 'models_saved/dt_model.joblib')
    
    return dt_model, dt_preds

def rf_classification(X_train, X_test, y_train, n_jobs=None, n_estimators=100, max_depth=None, random_state=None, save=False):

    rf_model = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators,max_depth=max_depth, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    if save:
        dump(rf_model, 'models_saved/rf_model.joblib')

    return rf_model, rf_preds

def xgb_classification(X_train, X_test, y_train, max_depth=0, eta=0, min_child_weight=0, gamma=0, save=False):
    
    if max_depth > 0:
        xgb_model = XGBClassifier(n_jobs=-1, max_depth=max_depth, eta=eta, min_child_weight=min_child_weight, gamma=gamma)
    else:
        xgb_model = XGBClassifier(n_jobs=-1)
    
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    if save:
        dump(xgb_model, 'models_saved/xgb_model.joblib')
    
    return xgb_model, xgb_preds

def dnn_classification(X_train, X_test, y_train, save=False):
    # Transformation en tableaux numpy, plus adapté pour le deep learning et pour SHAP ensuite
    X_train_array = X_train.to_numpy()
    X_train_array = X_train_array.astype(np.float32)
    X_test_array = X_test.to_numpy()
    X_test_array = X_test_array.astype(np.float32)

    seuil = .473 # taux de réussite moyen pour l'ensemble des joueurs
    negative_slope = .3

    dnn_model = Sequential()
    dnn_model.add(Input(shape=(X_train_array.shape[1],)))
    dnn_model.add(Dense(units=64))
    dnn_model.add(ReLU(negative_slope=negative_slope))
    dnn_model.add(Dense(units=128))
    dnn_model.add(ReLU(negative_slope=negative_slope))
    dnn_model.add(Dropout(rate=.2))
    dnn_model.add(Dense(units=1, activation='sigmoid'))

    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy',
                                             patience=5,
                                             mode='max',
                                             restore_best_weights=True)
    
    lr_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                             factor=0.5, 
                                             patience=2, 
                                             mode='min')
    
    dnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    dnn_model.fit(X_train_array, y_train, epochs = 30, batch_size = 64, validation_split = .2,
              callbacks = [early_stopping,lr_plateau], verbose=0) 
    
    y_prob_dnn = dnn_model.predict(X_test_array)
    dnn_preds = (y_prob_dnn > seuil).astype(int).flatten()

    if save:
        dnn_model.save('models_saved/dnn_model.h5')
    
    return dnn_model, dnn_preds

def classification_scores(y_test, y_preds):
    
    report = classification_report(y_test, y_preds, digits=3, output_dict=True)
    
    print(classification_report(y_test, y_preds, digits=3))
    print(pd.crosstab(y_test, y_preds, rownames=['Classe prédite'], colnames = ['Classe réelle']), "\n")
    
    accuracy = round(report['accuracy'], 3)
    f1_score_0 = round(report['0']['f1-score'], 3)
    f1_score_1 = round(report['1']['f1-score'], 3)
    
    return report, accuracy, f1_score_0, f1_score_1