from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import numpy as np
import pandas as pd
#from fancyimpute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# pip install --user fancyimpute 


# Définition d'une fonction de conversion en heures
def convert_to_hours(duree, unite):
    if unite == 'Jour(s)':
        return duree * 24
    elif unite == 'Minute(s)':
        return duree / 60
    elif unite == 'Semaine(s)':
        return duree * 168
    elif unite == 'Mois':
        return duree * 720
    elif unite == 'Année(s)':
        return duree * 8760
    elif unite == "Heure(s)":
        return duree
    elif unite == "non rapproché": # ça a l'aire d'être en minute, il faut une explication : 
        return duree/60

def preprocess_data(donnees):
    """Prétraite les données""" 
    
    data=donnees.copy()

    # Supprimer les doublons :    
    data.drop_duplicates(subset ="Identifiant fonctionnel du cas", keep = 'first', inplace=True)


    # Se débarasser des variables qui n'ont aucune importance 'théoriquement' : 
    data=data.drop(["CAP Référent","Code postal du cas","Date de l'exposition","Heure de l'exposition",
                    "Circonstance de l'exposition","Date du premier événement",
                    "Lieu de l'exposition","TERMCODE","Date de naissance",
                    "Identifiant fonctionnel du cas","PSS","Délai du SSM",
                    "Unité de durée de l'exposition à l'agent","Hiérarchie principale niveau 1",
                    "Contenu de la note du cas","Cause (hiérarchie niveau 1)","Cause (hiérarchie niveau 2)",
                    "Statut de la note du cas","Pentagramme du rédacteur de la note",
                    "Libellé agent", 
                    "TERMCODE","Symptôme - Syndrome - Maladie (SSM)"],axis=1) # "Lieu de prise en charge réalisé", "Évolution finale",
    """
    --> On supprime "Délai du SSM" mais on garde "Délai du SSM en min".
    --> On supprime "Circonstance d'exposition" mais on garde "Classe de la circonstance d'exposition".
    --> On supprime "PSS" et on garde "Gravité" parce que c'est la même chose et avec le même nombre de données manquantes.
    --> On supprime "Unité de durée de l'exposition à l'agent" parce qu'elle est une variable qui a une seule valeur : "non renseigné".
    --> On supprime "Hiérarchie principale niveau 1"  parce qu'elle est une variable qui a une seule valeur : "AGENT".
    --> On supprime la variable "Date du premier événement" parce que ça va rien chager dans notre étude, sinon il faut faire la différence avec mntns.
    --> Pour le moment, on supprime la variable "Contenu de la note du cas", mais on va le traiter séparément avec les outils "texte mining".
    --> On supprime les variables "Statut de la note du cas" et "Pentagramme du rédacteur de la note" qui servent juste à des choses administratives. 
    --> Entre "Référence fonctionnelle de l'agent d'exposition (avant remplacement)" et "Libellé agent", on garde le premier 
    parce qu'elles font référence à la même chose et on évite le libellé agent pour éviter les erreur d'ortographe.
    --> On supprime "Lieu de prise en charge réalisé" qui est une variable catégorielle ayant deux classes crai et faux.
    --> On supprime TERMCODE et Symptôme - Syndrome - Maladie (SSM) car elles sont corrélées à 100% avec Terme principale du SSM, et elles resprésentent tous la même chose. 
    --> On  a supprimé Cause (hiérarchie niveau 1 et 2) parce que après garder que les lieux de PEC vrai, elle contient que de données manquantes.
    """
    # garder que les individus avec lesquels le médecin ne se trompais pas !
    data = data[data["Lieu de prise en charge réalisé"] == 'Vrai']

    # Supprimer la colenne "Lieu de prise en charge réalisé" qui prends maintenant qu'une seule valeur.     
    data=data.drop("Lieu de prise en charge réalisé",axis=1) 
    
    """
    La 2eme stratégie :
    data.loc[data["Lieu de prise en charge réalisé"] =="Faux","Nom du lieu de prise en charge"] ="Cas d'urgence" 
    # Supprimer la colenne "Lieu de prise en charge réalisé". 
    data=data.drop("Lieu de prise en charge réalisé",axis=1)
    """
    
    # Remplacer les valeurs 'non renseigné' par nan
    data.replace('non renseigné', pd.np.nan, inplace=True)
    # Remplacer "Inconnu (âge)"  par nan 
    data.replace("Inconnu (âge)", pd.np.nan, inplace=True)
    # Remplacer les valeurs 'Inconnu' par nan
    data.replace(["Inconnu" , "Inconnue",'Sans objet'], pd.np.nan, inplace=True)
    
    
    # Exclure les données manquantes de la variable cible : "Lieu de prise en charge réalisé" et "Évolution finale"
    data = data.dropna(subset=["Nom du lieu de prise en charge"])

    
    
    # Garder que les observations qui ont Vrai comme valeur de 'RTU ou hors RTU'.
    data = data[data["RTU ou hors RTU"] == 'Vrai']
    # Ensuite supprimer cette variable qui contient maintenant une seule valeur 
    data=data.drop(["RTU ou hors RTU"],axis=1)
    
    
    # Exclure les observations qui prends la valeur Nulle en Imputabilité (globale).
    data = data[data["Imputabilité (globale)"] != 'Nulle']
    
    
    # Exclure les individus qui ont le statut : suivi jugé inutile
    data = data[data["Statut du cas"] != 'Classé : suivi jugé inutile']
    

    # Exclure les observations qui prends la valeur Nulle en Imputabilité (SSM).
    data = data[data["Imputabilité (SSM)"] != 'Nulle']


    # Exclure les animaux de l'étude : 
    data = data[data["Nature (exposé/concerné)"] != 'Animal']
    # Ensuite supprimer cette variable qui contient maintenant une seule valeur 
    data=data.drop(["Nature (exposé/concerné)"],axis=1)
    
    # Classer les valeurs de la variable cible Nom du lieu de prise en charge.
    """
    On va regrouper nos classes à 3 classes : 
    --> Domicile/Lieu d'exposition :Domicile, Lieu d'exposition, Foyer d'accueil / foyer de vie, EHPAD / maison de retraite médicalisée,
            Autre lieu.
    --> Cas d'urgence : SMUR, Hôpital / clinique : SAU / urgences, Cabinet ou urgences ophtalmologiques, 
            Hôpital / clinique : secteur d'hospitalisation, Hôpital / clinique : réanimation / soins intensifs, 
            Hôpital / clinique psychiatrique, Cabinet SOS médecin, Centre d'accueil et de permanence des soins (CAPS), 
            Cabinet vétérinaire, Hôpital / clinique : UHCD / HTCD, Hôpital / clinique : consultation externe, Cabinet dentaire,
            Hôpital / clinique : SSR. 
    --> Cas médical différé : Cabinet médical libéral, Pharmacie de ville, Infirmerie.
    """
    indice_colonne = data.columns.get_loc("Nom du lieu de prise en charge")

    for i in range(len(data)):
        if data.iloc[i,indice_colonne]=="SMUR" or data.iloc[i,indice_colonne]=="Hôpital / clinique : SAU / urgences" or data.iloc[i,indice_colonne]=="Cabinet ou urgences ophtalmologiques" or data.iloc[i,indice_colonne]=="Hôpital / clinique : secteur d'hospitalisation" or data.iloc[i,indice_colonne]=="Hôpital / clinique : réanimation / soins intensifs" or data.iloc[i,indice_colonne]=="Hôpital / clinique psychiatrique" or data.iloc[i,indice_colonne]=="Cabinet SOS médecin" or data.iloc[i,indice_colonne]=="Centre d'accueil et de permanence des soins (CAPS)" or data.iloc[i,indice_colonne]=="Cabinet vétérinaire" or data.iloc[i,indice_colonne]=="Hôpital / clinique : UHCD / HTCD" or data.iloc[i,indice_colonne]=="Hôpital / clinique : consultation externe" or data.iloc[i,indice_colonne]=="Cabinet dentaire" or data.iloc[i,indice_colonne]=="Hôpital / clinique : SSR" :
            data.iloc[i,indice_colonne]="établissement de santé"
        elif data.iloc[i,indice_colonne]=="Domicile" or data.iloc[i,indice_colonne]=="Lieu d'exposition" or data.iloc[i,indice_colonne]=="Foyer d'accueil / foyer de vie" or data.iloc[i,indice_colonne]=="EHPAD / maison de retraite médicalisée" or data.iloc[i,indice_colonne]=="Autre lieu":
            data.iloc[i,indice_colonne]="Domicile/lieu expo"
        elif data.iloc[i,indice_colonne]=="Cabinet médical libéral" or data.iloc[i,indice_colonne]=="Pharmacie de ville" or data.iloc[i,indice_colonne]=="Infirmerie":
            data.iloc[i,indice_colonne]="Cabines/clinique libéraux"
    

    # Diminuer le nombre de classes de la variable "Classe du lieu de l'exposition"
    """
    On va grouper nos classes à 5 grandes classes : domicile, école, maison de retraite, centre de santé et autre. 
    --> domicile : Habitation / Domicile, Collectivité (centre d'hébergement, foyer de vie…), 
    --> école : Ecole maternelle, Lycée, Collège, Ecole primaire, Crèche, Université / Enseignement supérieur, 
    --> maison de retraite : EHPAD / Maison de retraite.
    --> centre de santé : regardez dans la boucle en dessous XD.
    --> autre : autres.
    """
    indice_colonne = data.columns.get_loc("Classe du lieu de l'exposition")

    for i in range(len(data)):
        if data.iloc[i,indice_colonne]=="Habitation / Domicile" or data.iloc[i,indice_colonne]=="Collectivité (centre d'hébergement, foyer de vie…)" :
            data.iloc[i,indice_colonne]="domicile"
        elif data.iloc[i,indice_colonne]=="Ecole maternelle" or data.iloc[i,indice_colonne]=="Lycée" or data.iloc[i,indice_colonne]=="Collège" or data.iloc[i,indice_colonne]=="Ecole primaire" or data.iloc[i,indice_colonne]=="Crèche" or data.iloc[i,indice_colonne]=="Université / Enseignement supérieur" :
            data.iloc[i,indice_colonne]="école"
        elif data.iloc[i,indice_colonne]=="EHPAD / Maison de retraite" :
            data.iloc[i,indice_colonne]="maison de retraite"
        elif data.iloc[i,indice_colonne]=="Etablissement médico social (MAS / FAS / FAM / ESAT)" or data.iloc[i,indice_colonne]=="Hôpital général / clinique" or data.iloc[i,indice_colonne]=="Hôpital psychiatrique (CHS) / clinique psychiatrique / CMP" or data.iloc[i,indice_colonne]=="Laboratoire" or data.iloc[i,indice_colonne]=="Pharmacie de ville / officine" or data.iloc[i,indice_colonne]=="Cabinet medical de ville" or data.iloc[i,indice_colonne]=="Cabinet dentaire de ville" or data.iloc[i,indice_colonne]=="Clinique veterinaire" or data.iloc[i,indice_colonne]=="Hébergement / protection infantile (IME / ITEP)" :
            data.iloc[i,indice_colonne]="centre de santé"
        else :
            data.iloc[i,indice_colonne]="autre"

    # Remplacer Libellé antécédent niveau 1 et Libellé antécédent niveau 2 par antécédent :
    """
    Sachez que Libellé antécédent niveau 2 n'est qu'une précision de Libellé antécédent niveau 1. 
    On va remplacer Libellé antécédent niveau 1 et Libellé antécédent niveau 2 par "antécédent" qui est "oui" si on a 
    un antécédent (c_à_d si Libellé antécédent niveau 1 est nin nul) et "non" sinon. 
    """
    #print(data["Libellé antécédent niveau 1"].isnull().sum())
    indice_colonne = data.columns.get_loc('Libellé antécédent niveau 1')
    for i in range(len(data)):
        if pd.isna(data.iloc[i,indice_colonne])==True:
            data.iloc[i,indice_colonne]='non'
        else :
            data.iloc[i,indice_colonne]='oui'
    # Maintenant on change le nom de la colonne et on supprime la colonne Libellé antécédent niveau 2. 
    data = data.rename(columns={'Libellé antécédent niveau 1': 'antécédent'})
    data=data.drop(["Libellé antécédent niveau 2"],axis=1)

    
    # Diviser la variable "Exploration hiérarchie Snomed" en deux classes: "symptomatique" s'il y a des symptômes et "asymptômatique" sinon. 
    indice_colonne = data.columns.get_loc("Exploration hiérarchie Snomed")
    for i in range(len(data)):
        if data.iloc[i,indice_colonne]=="aucun symptôme constaté" :
            data.iloc[i,indice_colonne]="asymptômatique"
        elif pd.isna(data.iloc[i,indice_colonne])==True :
            data.iloc[i,indice_colonne]=data.iloc[i,indice_colonne]
        else :
            data.iloc[i,indice_colonne]="symptômatique"
            
    
    # diviser la variable Terme principal du SSM en deux classes : 
    """
    Si la variable prends la valeur "aucun symptôme constaté", alors on remplace cette dernière par 'non' sinon 
    on la remplace par 'oui'.
    """
    indice_colonne = data.columns.get_loc("Terme principal du SSM")
    for i in range(len(data)):
        if data.iloc[i,indice_colonne]=="aucun symptôme constaté" :
            data.iloc[i,indice_colonne]="non"
        elif pd.isna(data.iloc[i,indice_colonne])==True :
            data.iloc[i,indice_colonne]=data.iloc[i,indice_colonne]
        else :
            data.iloc[i,indice_colonne]="oui"
            
    
    """
    L'objectif de ces 3 prochaines lignes est de remplacer les deux colonnes "Délai de l'exposition" et "Unité de temps du délai de l'exposition" 
    par une colonne "Délai de l'exposition par heure" en rendant les valeurs de même unité. 
    """
    # Application de la fonction de conversion de l'heure à chaque valeur de durée
    data["Délai de l'exposition par heure"] = [convert_to_hours(duree, unite) for duree, unite in zip(data["Délai de l'exposition"], data["Unité de temps du délai de l'exposition"])]
    # Suppression de la colonne durée de l'exposition et unité :
    data = data.drop("Unité de temps du délai de l'exposition", axis=1)
    data = data.drop("Délai de l'exposition", axis=1)
    
    
    """
    il y a plusieurs variables numériques qui sont dans notre base de données sous forme d'un objet et il faut les trasformer à des floats, ces variables sont :
    Âge (en années), Poids (kg), Délai du SSM (min),Délai à l'exposition du lieu de prise en charge (min),Valeur quantitative de la dose d'exposition.                                                                                                                     
    """
    # Âge (en années) : remplacer la virgule par un point décimal
    data["Âge (en années)"]= data["Âge (en années)"].str.replace(',', '.')
    # Changer l'objet à float. 
    data["Âge (en années)"]= data["Âge (en années)"].astype(float)
        
    # Poids (kg) :
    data["Poids (kg)"]= data["Poids (kg)"].str.replace(',', '.')
    data["Poids (kg)"]= data["Poids (kg)"].astype(float)
    
    # Délai du SSM (min) :
    data["Délai du SSM (min)"]= data["Délai du SSM (min)"].str.replace(',', '.')
    # Enlecer l'espace insécable 
    data["Délai du SSM (min)"]=data["Délai du SSM (min)"].str.replace('\xa0','')
    # Remplacer l'espace pa le vide :
    data["Délai du SSM (min)"]=data["Délai du SSM (min)"].str.replace(' ','')
    # Changer l'objet à float. 
    data["Délai du SSM (min)"]= data["Délai du SSM (min)"].astype(float)
    
    
    # Délai à l'exposition du lieu de prise en charge (min) :
    data["Délai à l'exposition du lieu de prise en charge (min)"]= data["Délai à l'exposition du lieu de prise en charge (min)"].str.replace(',', '.')
    # Enlecer l'espace insécable 
    data["Délai à l'exposition du lieu de prise en charge (min)"]=data["Délai à l'exposition du lieu de prise en charge (min)"].str.replace('\xa0','')
    # Remplacer l'espace pa le vide :
    data["Délai à l'exposition du lieu de prise en charge (min)"]=data["Délai à l'exposition du lieu de prise en charge (min)"].str.replace(' ','')
    # Changer l'objet à float. 
    data["Délai à l'exposition du lieu de prise en charge (min)"]= data["Délai à l'exposition du lieu de prise en charge (min)"].astype(float)

    # Valeur quantitative de la dose d'exposition :
    data.loc[0,"Valeur quantitative de la dose d'exposition"]
    data["Valeur quantitative de la dose d'exposition"]= data["Valeur quantitative de la dose d'exposition"].str.replace(',', '.')
    # Changer l'objet à float. 
    data["Valeur quantitative de la dose d'exposition"]= data["Valeur quantitative de la dose d'exposition"].astype(float)
    tab=data.copy()
    print("les collonnes après le nettoyage sont : ",data.columns)


    # Récupération des noms des colonnes :  
    column_names = list(data.columns)

    # transformation des données vers array : 
    data=data.values
    # séparer les features catégorielles et les features numériques (Attention : il faut exclure 20 parce que c'est la target).
    col_cat=[0,1,2,4,7,8,9,10,11,12,13,14,15,16,17,29,30,32,33,34,35,36,37,38,40,41,20] # les colonnes catégorielles 
    col_num=[3,5,6,18,19,21,22,23,24,25,26,27,28,31,39,42] # les colonnes numériques
    X_cat = np.copy(data[:, col_cat]) 
    X_num = np.copy(data[:, col_num]) 
    
    # Récupération des noms des variables
    selected_columns = [column_names[i] for i in col_num + col_cat]  # Sélection des colonnes utilisées dans le modèle
    
    # Imputation multiple avec MICE de données manquantes numériques : 
    imputer_num = IterativeImputer(random_state=0)
    imputed_data_num = imputer_num.fit_transform(X_num)
    
    """
    # Initialisation de l'Imputer
    imputer = IterativeImputer(random_state=0)
    # Imputation des valeurs manquantes
    X_imputed = imputer.fit_transform(X)
    """
    
    # Imputation des variables catégorielles par mode : 
    X_cat_=pd.DataFrame(X_cat)
    data_imputed_cat = X_cat_.fillna(X_cat_.mode().iloc[0])
    
    # normaliser les variables numériques :
    Scaler=StandardScaler()
    Scaler.fit(imputed_data_num)
    X_num_normalisee=Scaler.transform(imputed_data_num) 
    # Transformer les variables catégorielles à des variables numériques :
    X_cat_bin = OneHotEncoder().fit_transform(data_imputed_cat).toarray()
    

    ######
    XX=np.concatenate((data_imputed_cat,imputed_data_num),axis=1)
    XX=pd.DataFrame(XX)
    ######

    # Features préparées : 
    X=np.concatenate((X_num_normalisee,X_cat_bin),axis=1)
    
    # Target (Nom du lieu de prise en charge) qui prends les valeurs Domicile/lieu expo,établissement de santé et Cabines/clinique libéraux:
    Y=data[:,20] # ou Y=donnees["Évolution finale"] avec donnees=data.copy() avant de faire data=data.values
    # transformer à une variable numérique :
    le = LabelEncoder()
    y=le.fit_transform(Y)
    
  
    # Séparation des données en ensembles de formation et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test #selected_columns,  XX #