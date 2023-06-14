#Importer les librairies nÃ©cessaires pour faire nos algorithmes
from sklearn.metrics import  make_scorer,accuracy_score, f1_score ,confusion_matrix #,precision_score
from sklearn.model_selection import cross_val_score,KFold,cross_val_predict
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Creation de notre propre score :
def custom_score(y_true, y_pred):
    
    """
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    # Extraire les composantes de la matrice de confusion
    L11,L12,L13,L14,L21,L22,L23,L24,L31,L32,L33,L34,L41,L42,L43,L44 = cm.ravel() # ligne par ligne 
    score1=L11/(L11+L12+L13+L14)
    score2=L11/(L21+L31+L41+L11)
    """
    f1=f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    return (accuracy+f1)/2
custom_scorer = make_scorer(custom_score)



kf = KFold(n_splits=10, shuffle=True, random_state=0)
def run_classifiers(clfs,Xtrain,Ytrain):
    scores = []
    for i in clfs:
        clf = clfs[i]
        scores_custom = cross_val_score(clf, Xtrain, Ytrain, cv=kf, scoring=custom_scorer)
        #scores_custom = cross_val_score(clf, Xtrain, Ytrain, cv=kf, scoring=lambda y_true, y_pred: custom_score(y_true, y_pred, labels=[0, 1, 2, 3]))
        #scores_precision = cross_val_score(clf, Xtrain, Ytrain, cv=kf, scoring='precision')
        score = np.mean(scores_custom)
        scores.append(score)
        print("\n \n the score for {0} is: {1:.3f}".format(i, score))
        #matrice_confusion=confusion_matrix(Xtrain, Ytrain)        
        #print("\n la matrice de confusion est :", matrice_confusion)
        
        # Calculer la matrice de confusion pendant la validation :
        y_pred_val = cross_val_predict(clf, Xtrain, Ytrain, cv=kf)
        cm = confusion_matrix(Ytrain, y_pred_val, labels=[0, 1, 2, 3])
        print("\nLa matrice de confusion pour {0} lors de la validation est :".format(i))
        print(cm)
        
        # Afficher la matrice de confusion avec des couleurs
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matrice de confusion pour {0}".format(i))
        plt.xlabel("Valeurs prédites")
        plt.ylabel("Valeurs réelles")
        plt.show()

    best_clf_idx = np.argmax(scores)
    print("\nThe best classifier is {0} with a score of {1:.3f}".format(list(clfs.keys())[best_clf_idx], scores[best_clf_idx]))


def train_model(model,X_train, y_train):
    #Entraine le modele
    model.fit(X_train, y_train)
    return model





# Amelioration de notre modele en optimisant les hypeparametres du modele : 
def improve_params(model,X_train, y_train,param_grid,):
    grid=GridSearchCV(model,param_grid,cv=kf,scoring='custom_scorer')
    grid.fit(X_train,y_train)
    improved_model=grid.best_estimator_
    improved_model.fit(X_train, y_train)
    return improved_model



# Voir le score du modele : 
def evaluate_model(model, X_test, y_test):
    """ fonction pour faire l'évaluation du modèle"""
    y_pred = model.predict(X_test)
    score = custom_score(y_test, y_pred)
    print('Le score de notre modèle est :', score)

def predict(model, house_features):
    #Fait une prediction
    predicted_price = model.predict([house_features])
    return predicted_price[0]

