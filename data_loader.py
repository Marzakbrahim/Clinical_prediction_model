import pandas as pd

def load_data(file_path):
    """Charge les données depuis un fichier CSV"""
    data = pd.read_csv(file_path, delimiter=',', skiprows=3)
    return data
