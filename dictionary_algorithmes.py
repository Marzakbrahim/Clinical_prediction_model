"""
Si on veut utiliser un autre algorithme machine learning qui n'existe pas dans 
notre dictionnaire clfs, on l'ajoute.
"""


#Importer les librairies nécessaires pour faire nos algorithmes
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier , AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


clfs = {
'LR': LogisticRegression(),
'NBS': GaussianNB(),
'RF': RandomForestClassifier(n_estimators=200, random_state=1),
'KNN': KNeighborsClassifier(n_neighbors=10),
'MLP': MLPClassifier(hidden_layer_sizes=(20,10), random_state=1),
'xgboost': XGBClassifier(),
'BGC': BaggingClassifier(n_estimators=200, random_state=1),
'AB': AdaBoostClassifier(n_estimators=200, random_state=1),
'AC': DecisionTreeClassifier(criterion='gini',random_state=1),
'AID3': DecisionTreeClassifier(criterion='entropy',random_state=1)
}