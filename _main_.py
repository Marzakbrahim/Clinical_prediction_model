import sys
sys.path.append('W:/documets stage/code_projet/data_loader.py') 
from  data_loader import load_data
from data_preprocessing import preprocess_data
from dictionary_algorithmes import clfs
from model import train_model, evaluate_model, predict,run_classifiers,custom_score,improve_params,CustomScorer
from sklearn.metrics import  make_scorer #,confusion_matrixf1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

# Chargement des données
data = load_data("W:/documets stage/Rapport 1.csv")

# Prétraitement des données
X_train, y_train, X_test, y_test= preprocess_data(data)   #  tab_sas

#tab_sas.to_excel('W:/documets stage/tab_sas.xlsx', index=False)

# choisir le meilleur algorithme :
run_classifiers(clfs,X_train, y_train)

# Entraînement du modèle
model=XGBClassifier()
model=train_model(model,X_train, y_train)

# Affichage des hyperparamètres et de leurs valeurs pour les optimiser :
hyperparameters = model.get_params()
for param, value in hyperparameters.items():
    print(param, "=", value)

######################################################### Sélection de variables : #####################################################
# peut être j'envisage faire label encoder au lieu de onehot pour résoudre le problème     
importance = model.feature_importances_

features = [f"Feature {i}" for i in range(len(importance))]

# Trier les importances et les fonctionnalités correspondantes
sorted_importances, sorted_features = zip(*sorted(zip(importance, features), reverse=True))

# Afficher les importances triées de manière croissante
for imp, feature in zip(sorted_importances, sorted_features):
    print(f"{feature}: Importance {imp}")
########################################################################################################################################

# choisir les meilleurs hyperparamètres : 
custom_scorer = make_scorer(custom_score)
from sklearn.model_selection import GridSearchCV
param_grid={
    'learning_rate': [1.0,0.9,0.5,0.4,0.300000012,0.1, 0.01, 0.001],
    'n_estimators': [50,70,100,300, 500,700, 1000,10000],
    'max_depth': [3,4,5,6,7],
    'subsample': [0.8,0.9, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
grid=GridSearchCV(model,param_grid,cv=5,scoring=custom_scorer)
grid.fit(X_train,y_train)
print(grid.best_params_)
model=grid.best_estimator_
model.fit(X_train,y_train)

########################################################################################################################################

# Évaluation du modèle
evaluate_model(model, X_test, y_test)

"""
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.01, verbose=0,
                   warm_start=False)
"""

"""
print(data["Imputabilité (globale)"].isnull().mean())
data["Nom du lieu de prise en charge"].unique()
print(data["Imputabilité (globale)"].isnull().mean())
model=RandomForestClassifier(n_estimators=200, random_state=1)
our_mod=train_model(model,X_train, y_train)
y_pred = our_mod.predict(X_test)
# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
data.columns
"""
data["Contenu du note"]