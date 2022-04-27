import pandas as pd
import numpy as np
from pathlib import Path
import os 
import yaml
import pickle

#Parameter einlesen
params = yaml.safe_load(open("params.yaml"))["trainieren"]
#modell = params['modell']

#importiere modell
#from sklearn.tree import modell
from sklearn.ensemble import RandomForestClassifier

#Daten einlesen
x_train =pd.read_csv("funktionalisieren/train/x_train.csv", sep=",")
y_train =pd.read_csv("funktionalisieren/train/y_train.csv", sep=",")


#Parameters fürs Modell
#max_leaf_nodes=params['max_leaf_nodes']
#random_state  =params['random_state ']
n_estimators  =params['n_schaetzer']  
#Modell Training
modell = RandomForestClassifier(n_estimators)
modell.fit(x_train, y_train)

#Modell speichern
data_path= os.path.join("Modell")
os.makedirs(data_path, exist_ok=True)

filename = 'Modell/modell_ansatz2.sav'
pickle.dump(modell, open(filename, 'wb'))
#pickle.dump(data_path/modell_ansatz2)