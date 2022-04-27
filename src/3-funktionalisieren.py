from genericpath import exists
import pandas as pd
import numpy as np
from pathlib import Path
import os 
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale #mean 0, s 1
import math
import sys
import glob 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

#-Signalauswahl - rekursive feauture elimination
from sklearn.feature_selection import RFE
#print('Ansatz 2')
params = yaml.safe_load(open("params.yaml"))["funktionalisieren"]
#modell= params['modell'] # modell
#from sklearn.tree import  modell
#from sklearn.tree import  params['modell']
from sklearn.ensemble import RandomForestClassifier

#clf=RandomForestClassifier(n_estimators=100) #gaussian classifier
#clf.fit(X_train, y_train)

# Scale data, mean 0 , s 1
#daTa = scale(data )

def sauber_datensatz(df):
    '''valide Datensatz'''
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

dframe_zu_kombinieren=pd.read_csv("Aufbereiten/dframe_zu_kombinieren.csv" , sep=",")

#dframe_zu_kombinieren=sauber_datensatz(dframe_zu_kombinieren)

dframe_zu_kombinieren.replace([np.inf, -np.inf], np.nan, inplace=True)
dframe_zu_kombinieren.fillna(0, inplace=True)

#folder, um datei zu speichern
data_path = os.path.join("funktionalisieren","train")
data_path1= os.path.join("funktionalisieren","test")

os.makedirs(data_path, exist_ok=True)
os.makedirs(data_path1, exist_ok=True)

#data_ansatz2_2.to_csv("data/aufbereitet/export.csv", index=False)

###-----Daten Labeln: es wird versucht, Cluster in den Daten zu finden-----####
X=dframe_zu_kombinieren.copy()
scaler = MinMaxScaler()
scaler.fit(X)
X= scaler.transform(X)
#inertia = []
sse = {}
###----Finde Beste Cluster----####
for i in range(1,round(len(list(dframe_zu_kombinieren.columns))/2)):
    kmeans = KMeans(n_clusters=i, init="k-means++",n_init=10, tol=1e-04, random_state=42)
    kmeans.fit(X)
    #clusters['label'] = kmeans.labels_
    #inertia.append(kmeans.inertia_)
    sse[i]=kmeans.inertia_ #Inertia: summe Abstände der Datenpunkten zu ihrem nächstgelegenen Clusterzentrum

#scaler.inverse_transform(X)[:, [0]]

clusters = pd.DataFrame(X, columns=dframe_zu_kombinieren.columns) #dt_1_sek.columns
clusters['label'] = kmeans.labels_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Anzahl Cluster")
plt.ylabel("SSE")
plt.show()

number=False
while not number:
    anzahl=int(input("Bitte geben Sie Anzahl Cluster an:"))
    if type(anzahl)==int:
      kmeans = KMeans(n_clusters=anzahl, init="k-means++",n_init=10, tol=1e-04, random_state=42)
      kmeans.fit(X)
      number=True
    else:
        print('Bitte geben sie eine Zahl ein')

y = kmeans.fit_predict(X)

#Rekursiv Feature Elimination - selektiere  beste Signale aus
n_estimators=params['n_schaetzer']
modelle =RandomForestClassifier(n_estimators) #max_leaf_nodes=10, random_state=0

#Define RFE  - RFE(modelle, 5)
rfe = RFE(modelle)
#benutze RFE und selektiere  top (5) (features) Parameters 
fit = rfe.fit(dframe_zu_kombinieren, y)

feature_names = list(dframe_zu_kombinieren.columns)

#Create a dataframe for the results 
df_RFE_results = []
for i in range(dframe_zu_kombinieren.shape[1]):
    df_RFE_results.append(
        {        #data_ansatz2_2.feature_names[i]
            'Signal_namen': feature_names[i],
            'Selektiert':  rfe.support_[i],
            'RFE_ranking':  rfe.ranking_[i],
        }
    )

df_RFE_results = pd.DataFrame(df_RFE_results)
df_RFE_results.index.name='Columns'

#Selektiere nur Relevante Signale fürs Modell
col_true= list(df_RFE_results['Signal_namen'].loc[df_RFE_results['Selektiert'] == True])
print('Beste Signale\n', col_true)

#es wird nun mit den Besten Signale gearbeitet
beste_signal=dframe_zu_kombinieren[col_true]

#Daten aufteilen
#independent
#x_ansatz2 = data_ansatz2_2.loc[:, data_ansatz2_2.columns != 'result']
#target dependent Variable
#y_ansatz2 = data_ansatz2_2[['result']].copy()

#test_size aus split
testSize = params['split']

x_train2, x_test2 , Y_train2, Y_test2 = \
    train_test_split(beste_signal,y, test_size=testSize, random_state=325)

#Die Daten speichern
pd.DataFrame(x_train2).to_csv("funktionalisieren/train/x_train.csv", index=False)
pd.DataFrame(Y_train2).to_csv("funktionalisieren/train/y_train.csv", index=False)
pd.DataFrame(x_test2).to_csv("funktionalisieren/test/x_test.csv",  index=False)
pd.DataFrame(Y_test2).to_csv("funktionalisieren/test/y_test.csv", index=False)