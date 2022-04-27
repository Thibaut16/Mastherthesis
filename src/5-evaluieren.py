from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import os 
import yaml
import pickle
import json

x_test  =pd.read_csv("funktionalisieren/test/x_test.csv", sep=",")
y_test  =pd.read_csv("funktionalisieren/test/y_test.csv", sep=",")

#Modell hochladen
filename = 'Modell/modell_ansatz2.sav'
modell_ansatz2= pickle.load(open(filename, 'rb'))
#modell_ansatz2= pickle.loads("Modell/modell_ansatz2.pckl")

#Modell evaluation (validierung)  

Y_predicted_2 = modell_ansatz2.predict(x_test)
Y_predicted_proba = modell_ansatz2.predict_proba(x_test)

print(accuracy_score( y_test , Y_predicted_2) *100)
print(confusion_matrix( y_test , Y_predicted_2))


data = {"accuracy_score":"99.89733059548254"}

target_names = ['class 0', 'class 1', 'class 2']#, 'class 3'
print(classification_report(y_test, Y_predicted_2, target_names=target_names))
print('metriken speichern')
data_path= os.path.join("Metrik")
os.makedirs(data_path, exist_ok=True)

with open('Metrik/metrik.json','w') as d:
  json.dump(data,d)

# time,BR5_Bremsdruck[Unit_Bar],ESP_Fahrer_bremst[],ACC_Minimale_Bremsung_valid[1],
# BSL_ZAS_Kl_15_valid[1],MO1_Drehzahl_valid[1],
# MO_Gangposition[],LWI_Lenkradwinkel[Unit_DegreOfArc],
# D_LWI_Lenkradw_Geschw[Unit_DegreOfArcPerSecon],
# KBI_Kilometerstand_valid[1],
# KBI_Handbremse_valid[1],
# KBI_angez_Geschw_valid[1],MO_StartStopp_StoppVorbereitung[],
# MO_StartStopp_StoppVorbereitung_valid[1],
# MO_Gangposition_valid[1],BSL_ZAS_Kl_15[],
# V_UTCTime_continous[1],ACC_Minimale_Bremsung[],
# BR5_Bremsdruck_valid[1],ESP_Fahrer_bremst_valid[1],
# GPS fix[1],KBI_angez_Geschw[Unit_KiloMeterPerHour],
# LWI_Lenkradwinkel_valid[1],D_LWI_Lenkradw_Geschw_valid[1],
# MO1_Drehzahl[Unit_MinutInver],KBI_Handbremse[],KBI_Kilometerstand[Unit_KiloMeter]
# ,MO1_Drehzahl,Kilometerstand,Kl_15[],Geschw[Unit_KiloMeterPerHour]
# f = open('Metrik/metrik.json',)  
# data = json.load(f) 
# type(data)
