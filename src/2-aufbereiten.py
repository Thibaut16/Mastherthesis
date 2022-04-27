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
import matplotlib as plt

#-Signalauswahl - rekursive feauture elimination
from sklearn.feature_selection import RFE


#print('Ansatz 2')

#dic =  [ [1,np.nan], [2,2], [3,np.nan],[6,7],[np.nan,2] ]
#data = df_ana.copy()
#data = df_ana.copy()#  pd.DataFrame(dic, columns= ['A','B'])

#df  = pd.read_csv("data/extrahiert/export.csv", sep=",")
#print('here',data['KBI_angez_Geschw[Unit_KiloMeterPerHour]'].unique())

def findmin(alist):
    '''
    find max in case of nan in list
    '''
    findmin_=[]
    for val in alist:
        if not np.isnan(val):
            findmin_.append(val)
    min_ist=min(findmin_) 
    return min_ist

def check_firstvalids_are_nan(some_list):
    '''
    ueberpruefe, ob alle Werte in erste valid index nan ist
    wenn ja soll man abbrechen, denn es kann angenommen werden, dass man den Datensatz 
    vollstaendig bearbeitet hat 
    '''
    gib_es_valid=0
    for val in some_list:
        #print('!',some_list, val , 'len:', len(some_list))
        if np.isnan(val) or val is None:
            val= 0
        gib_es_valid = gib_es_valid+ int(val)
    return gib_es_valid>0 #True-> nicht nur nans

# def datenreihen_erstellen(Data):
#     '''
#     Diese Methode erstellt datenreihen für einen Datensatz 
#     '''
#     import math
# def convert_to_sekunden(x):
#     return x/1000
#time=df[['time']]
#df['time']= df['time'].apply(convert_to_sekunden)
# new_Df= df[df['time']<=1.000] #new_Df= df[df['time']<=1]
# test= df[df['time']>1.000]

#man liest die Datensätze nacheinander und kombiniert dieses zu einem
path = r'C:/Users/I008470/Documents/arbeit/masterarbeit_code/jordis-masterthesis/evaluate/'
all_daten= glob.glob(path+"/*.csv")
li=[]
for dateiname in all_daten:
    df = pd.read_csv(dateiname, index_col=None, header=0)
    
    if 'time' in df.columns:
      letzte_zeit_eintrag=df['time'].iloc[-1]
    #    num_in_df =math.floor(letzte_zeit_eintrag/1.000)
    #    rest = (letzte_zeit_eintrag/1.000)% num_in_df
      def seconds(beste_abtastRat):
                beste_abtastRate = []
                for i in beste_abtastRat:
                    i /= 1000.
                    #i=i+0.001
                    #i=format(i, '.3f') 
                    beste_abtastRate.append(float(i))
                return beste_abtastRate

      letzte_zeit_eintrag=df['time'].iloc[-1]
      avg_time= math.floor(letzte_zeit_eintrag/2)

      if avg_time>=1.000:
        #beste_abtastRate = [1.000*i for i in range(0,avg_time,1000) if i!=0]
        beste_abtastRat = [i*1.0 for i in range(0,avg_time,1000) if i!=0]
        beste_abtastRate = seconds(beste_abtastRat)
      else:
        beste_abtastRat = [i*1.0 for i in range(0,avg_time,(avg_time/2)) if i!=0]
        beste_abtastRate = seconds(beste_abtastRat)

      print('beste_abtastRate:',beste_abtastRate)

      columns= [col for col in df.columns if col!='time']
      result={}
      for abtast in beste_abtastRate:
            num_in_df =math.floor(letzte_zeit_eintrag/abtast)
            #print(letzte_zeit_eintrag,num_in_df, abtast)
            rest = (letzte_zeit_eintrag/abtast)% num_in_df
            test = [abtast*i for i in range(0,num_in_df,1000) if i!=0]
            test= seconds(test)
            test.append(letzte_zeit_eintrag)
            #print(test)
            #test.reverse()
            dTf= df.copy()
            sum_haeufig=0 
            dic_signal={}
            for indx, elem in enumerate(test):
                #dTf.loc[dTf['time']<=val, ['zeitfenster']] = indx
                #print(columns)
                for col in columns:
                    #print('-',col,indx, elem)
                    if col!='time' and indx==0:
                        tmp= df[df['time']<=elem]
                        tmp= tmp[tmp[col].notna()]
                        #tmp=tmp.dropna()
                        #c= tmp.count().sum()
                        #print('+',col,tmp[col].shape[0], 'c:',c)
                        if tmp[col].shape[0]==0:
                          dic_signal[col]= dic_signal.get(col, 0) + 1 #count zero
                        
                    if col!='time' and indx!=0:
                        tmp= df[df['time']<=elem]
                        Tmp= tmp[tmp['time']>test[indx-1]]
                        Tmp= Tmp[Tmp[col].notna()]

                        if Tmp[col].shape[0]==0:
                           dic_signal[col]= dic_signal.get(col, 0) + 1 
                        #print('-',col,Tmp[col].shape[0], 'c:',c)
                            
            #selektiere Signale mit Häufigkeiten >=Hälte von Test (aufgeteilten milli Sekunde)
            signal_bad=[k for k,v in dic_signal.items() if (v) >= len(beste_abtastRate)]
            all_signals=[sig for sig in df.columns if sig!='time']
            new_list = [sig for sig in all_signals if sig not in signal_bad]
            #print(signal_bad)
            #break
            result[abtast]=new_list   

        ### Die Abtastraten wurden überprüft und werden Jetzt dem Nutzer vorgeschlagen
      select_int={}
      for k in result.keys():
            select_int[k]=len(result[k])
            select_int

      beenden=False
      while not beenden:
            print('die angeschauten Sekunde sind:', select_int.keys())
            print('die Anzahl der auszuwählenden Signale, sollte man Datenzeile bilden wollen:'\
                , select_int.values())
            choice=float(input("selektiere eine Sekunde für das Daten sampling"))
            print(select_int[choice])
            beenden=input("sind Sie zufrieden mit der Wahl, geben sie True an sonst False")

      dt_filtered = df[result[choice]] #selektiere die Spalten

      ##--Datensatzt wird mit der ausgewählten Abtastrate gruppiert--##
      letzte_zeit_eintrag=df['time'].iloc[-1]
      num_in_df =math.floor(letzte_zeit_eintrag/choice)#letzte_zeit_eintrag/1.000
      rest = (letzte_zeit_eintrag/choice)% num_in_df
      test = [1.000*i for i in range(num_in_df) if i!=0]
      test.append(letzte_zeit_eintrag)
      test.reverse()
      dTf= df.copy()
      for indx, val in enumerate(test):
            dTf.loc[dTf['time']<=val, ['zeitfenster']] = indx

      group=dTf.groupby('zeitfenster')

      dt_1_sek = group.mean()
      dt_1_sek = dt_1_sek.reset_index()

      dt_1_sek=dt_1_sek.sort_values('zeitfenster', ascending=False)
      dt_1_sek = dt_1_sek.fillna(0)
      colum_filter=[col for col in dTf.columns if col!='zeitfenster']
      dt_1_sek = dt_1_sek[colum_filter]
      ###
      #Danach bildet man, den Datensatz pro Sekunde : data sample
      ###
        
      # num_in_df =math.floor(letzte_zeit_eintrag/1.000)
      # rest = (letzte_zeit_eintrag/1.000)% num_in_df
      # test = [1.000*i for i in range(num_in_df) if i!=0]
      # test.append(letzte_zeit_eintrag)
      # test.reverse()
      #df = df[signal]
      dTf= dt_1_sek.copy()

      # for indx, val in enumerate(test):
      #     dTf.loc[dTf['time']<=val, ['zeitfenster']] = indx

      # group=dTf.groupby('zeitfenster')
      # dt_1_sek= group.agg([np.mean])
      # dt_1_sek=dt_1_sek.sort_values('zeitfenster', ascending=False)

      ##--überprüfe Name und ändere diesen--##
      if os.path.exists('feature_store/signal-name.txt'):
            #falls es existiert, signal hierausholen
            with open('feature_store/signal-name.txt') as file:
              '''lese alle Signale und Deskriptive Werte'''
              lines = file.readlines()
              lines =[ line.rstrip() for line in lines]
            for signalColumn in lines:
                signalCol = signalColumn.split(",")
                col_behalt_in_df,col_vergleich=signalCol[0], signalCol[1]
                if (col_behalt_in_df in dt_1_sek.columns()) or (col_vergleich in dt_1_sek.columns()):
                 dt_1_sek.rename(columns={col_vergleich:col_behalt_in_df},inplace=True)
                

        #speichere, der gerade, bearbeitete datensatz
      li.append(dt_1_sek)
    else:
        print('Signal Zeit nicht vorhanden!!!','\n', dateiname)
        #sys.exit(1)
        continue

dframe_zu_kombinieren = pd.concat(li, axis=0, ignore_index=True)
# #test
# # num_in_df =math.floor(letzte_zeit_eintrag/0.250)
# # rest = (letzte_zeit_eintrag/0.250)% num_in_df

# print(df['time'].head())
# print('letzte_zeit_eintrag:',letzte_zeit_eintrag,\
# '1000sek in letzte_zeit_eintrag:', num_in_df, 'rest:',rest)

# data fuer 1 sekunde erstellen
# cols = [col for col in df.columns if col!= 'time']

# dic=[]
# def haufigkeit_sek(indx, elem,test):
#     '''
#     diese Methode zaehlt die Haeufigkeit pro Sekunde und hilft dabei einen neuen
#     df zu erstellen pro Sekunde
#     '''
#     rel_hauf_sek=[]
#     signal = []
#     data_pro_sekunde = [] 
#     for col in df.columns:
#         if col!='time' and indx==0:
#             tmp= df[df['time']<=elem]
    #         tmp= tmp[[col]]
    #         tmp=tmp.dropna()
    #         shape_=tmp[col].shape[0]
    #         rel_hauf_sek.append(shape_)
    #         signal.append(col)
    #     if col!='time' and indx!=0:
    #         tmp= df[df['time']<=elem]
    #         Tmp= tmp[tmp['time']>test[indx-1]]
    #         Tmp= Tmp[[col]]
    #         Tmp=Tmp.dropna()
    #         shapE_ =Tmp[col].shape[0]
    #         rel_hauf_sek.append(shapE_)
    #         signal.append(col)
    #  #um neuen df zu erstellen pro Sekunde
    # return rel_hauf_sek, signal
#DIC = {}
#test = [1.000*i for i in range(df) if i!=0] #1 sekunde (1.000 milli sekunden)
# #test = [0.250*i for i in range(num_in_df) if i!=0]
#test.append(test[-1]+rest) # rest der Zeit

#print(test, len(test))
#  haufigkeit_sek() wird nicht aufgerufen, falls  mindestens einen Signal in Text Datei geschrieben wurde
###

# suche = params['suche']
# if suche:
#     #signal= haufigkeit_sek()
#     for indx, elem in enumerate(test):
        
#       haufig, signal= haufigkeit_sek( indx, elem, test)
#       DIC[indx]=haufig
#       if indx==(len(test)-1):
#          DIC[indx+1]=signal

    # cols=list(new_Df.columns)
    # cols.remove('time')
    # print(haufig, new_Df.columns)
    # diC= {'Haufig_1sek':haufig, 'signal':signal}
    #     #dt = pd.DataFrame(mean_signale,signale, columns=['A','col'])
    # dt=  pd.DataFrame(diC)


#     dt=  pd.DataFrame(DIC)
#     # add col: dt[col1] + dt[col2]
#     dT = dt[[i for i in range(len(test)+1)]]
#     dT['zero_counts']=(dT==0).sum(axis=1)
#     dT.loc[dT['zero_counts']>=math.floor((len(test)/2)+3), [len(test)]] = 0
#     valid_cols = dT[[len(test)]]
#     valid_cols= valid_cols[valid_cols[len(test)]!=0]
#     validColumns = list(valid_cols[len(test)])
#     signal = validColumns
#     print(validColumns)

# else:
#     signal=list(df.columns)

###
#es wird Signale nicht selektiert, die in einem Sekunde (1000 milli Sekunden) nicht einmal aufgetreten sind
###

# new_Df= df[df['time']<=1.000] #new_Df= df[df['time']<=1]
# test= df[df['time']>1.000]
# def haufigkeit_sek():
#     rel_hauf_sek=[]
#     signal = []
#     for col in new_Df.columns:
#         if col!='time':
#             tmp= new_Df[[col]]
#             #tmp= tmp[[col]]
#             tmp=tmp.dropna()
#             rel_hauf_sek.append(tmp[col].shape[0])
#             signal.append(col)
#     result=[]
#     for indx, num in enumerate(rel_hauf_sek):
#         if num==0:
#             result.append(signal[indx])
#     return result

###





# dT #dt

# # data für 1 sekunde : 1000 milli sekunde
# data = pd.DataFrame(dic,columns=cols)

#Annaheme: man geht davon aus, dass fuer die ersten 1000 Sekunden,
# Signale, die einen Haeufigkeit weniger als 1 (also gar nicht auftauchen) 
# in diesem Zeitfenster keine Betrachtung finden und ausfallen 

#Spärlichkeit und fehlende Werte in dem Datensatz
# print('laenge vom Datensatz pro sekunde:', data.shape[0])
# fehlt=data.isnull().sum()
# nicht_nan_signal = []
# for num in list(fehlt):
#     nicht_nan_signal.append(data.shape[0]-num)
# print(nicht_nan_signal)
# print('Nan Werte pro Signal:\n')
# print(fehlt)
# print('die Obige Rechnung ergibt null, weil anstatt von nan,\
#     in data für den Mittelwert pro Sekunde (1000 milli sekunde)\
#     0 eingefügt wurde')

#signale auswaehlen

# columnS= []
# for signal in data.columns:
#     if data[signal].iloc[0]!=0:
#         columnS.append(signal)
# data = data[columnS] #Signale entfernt

#BSL erzeugen 
# clean_data=df_model.copy()    #loc- takes row , column as args
# clean_data.loc[df_model['KBI_angez_Geschw[Unit_KiloMeterPerHour]']==0,
#                      'BSL_ZAS_Kl_15[]']=0
# clean_data.loc[df_model['KBI_angez_Geschw[Unit_KiloMeterPerHour]']>0,
#                      'BSL_ZAS_Kl_15[]']=1
# clean_data

# #find Interval fuer die Klassifizierung
# def findmean(alist):
#     n= len(alist)
#     sum = 0
#     for i in alist:
#         sum =sum + i
#     if sum-n==0:
#         return 0
#     else:
#        return sum//n

# stadt, land, autobahn,abstand = [],[],[],[]

# def bereichserkennung():
#     '''
#     Diese Methode erkennt Bereiche, für die Einteilung 
#     sucht die Geschwindigkeit heraus
#     '''
#     col= False
#     if set(["KBI_angez_Geschw[Unit_KiloMeterPerHour]"]).issubset(list(data.columns)):
#        #DF= pd.DataFrame()
#        #DF['KBI_angez_Geschw[Unit_KiloMeterPerHour]']= \
#        DF=   data[['KBI_angez_Geschw[Unit_KiloMeterPerHour]']].copy()
#        col= True
#     else:
#         my_file = Path("test.txt")
#         try:
#             my_abs_path = my_file.resolve(strict=True)
#         except FileNotFoundError:
#             # doesn't exist
#             print('false')
#             exit()
#         else:
#             # exists
#             f = open("signal.txt", "r")
#             ges=str(f.read())
#             DF= data[ges].copy()
#     DF= DF.dropna()
#     if col:
#       values = list(DF['KBI_angez_Geschw[Unit_KiloMeterPerHour]'])
#     else:
#       values = list(DF[ges]) #suche, falls Signal einen anderen Namen hat
    
#     for i in range(len(values)):
#         if i+1<=len(values)-1:
#             abstand.append(values[i+1]-values[i])
#         if values[i]>=50 and values[i]<=65:
#             stadt.append(values[i])
#         if values[i]>=100 and values[i]<=120 :
#            land.append(values[i])

# bereichserkennung()
# a, b= findmean(stadt),findmean(land) 

# def signal_filtern(df):
#     '''
#     diese Methode entfernt Signale, die nicht so häufig im Interval vorkam
#     '''
#     col_auswahl=list(df.columns)
#     for col in df.columns:
#         if df[col].count()<=400: #Annahme 400 als gute Abtastrate fürs Zeitfenster 
#             col_auswahl.remove(col)#del df[col]
#     return col_auswahl 

# signalwahl= signal_filtern(data)
# data_new= data[signalwahl]

# data_ansatz2_2 = datenreihen_erstellen(data_new)

# #Ausgabe Labeling
# data_ansatz2_2['result']= (data_ansatz2_2['KBI_angez_Geschw[Unit_KiloMeterPerHour]'] > b)*2
# data_ansatz2_2.loc[data_ansatz2_2['KBI_angez_Geschw[Unit_KiloMeterPerHour]']<b+1,
#                      'result']=1
# data_ansatz2_2.loc[data_ansatz2_2['KBI_angez_Geschw[Unit_KiloMeterPerHour]']<a+1,
#                      'result']=0



# Scale data, mean 0 , s 1
#daTa = scale(data )

#folder, um datei zu speichern
data_path = os.path.join("Aufbereiten")
#data_path1= os.path.join("Aufbereiten","test")

os.makedirs(data_path, exist_ok=True)
#os.makedirs(data_path1, exist_ok=True)

dframe_zu_kombinieren.to_csv("Aufbereiten/dframe_zu_kombinieren.csv", index=False)

###-----Daten Labeln: es wird versucht, Cluster in den Daten zu finden-----####
# X=dframe_zu_kombinieren.copy()
# scaler = MinMaxScaler()
# scaler.fit(X)
# X= scaler.transform(X)
# #inertia = []
# sse = {}
# ###----Finde Beste Cluster----####
# for i in range(1,round(len(list(dframe_zu_kombinieren.columns))/2)):
#     kmeans = KMeans(n_clusters=i, init="k-means++",n_init=10, tol=1e-04, random_state=42)
#     kmeans.fit(X)
#     #clusters['label'] = kmeans.labels_
#     #inertia.append(kmeans.inertia_)
#     sse[i]=kmeans.inertia_ #Inertia: summe Abstände der Datenpunkten zu ihrem nächstgelegenen Clusterzentrum

# #scaler.inverse_transform(X)[:, [0]]

# clusters = pd.DataFrame(X, columns=dt_1_sek.columns)
# clusters['label'] = kmeans.labels_

# plt.figure()
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel("Anzahl Cluster")
# plt.ylabel("SSE")
# plt.show()

# number=False
# while not number:
#     anzahl=int(input("Bitte geben Sie Anzahl Cluster an:"))
#     if type(anzahl)==int:
#       kmeans = KMeans(n_clusters=anzahl, init="k-means++",n_init=10, tol=1e-04, random_state=42)
#       kmeans.fit(X)
#       number=True
#     else:
#         print('Bitte geben sie eine Zahl ein')

# y = kmeans.fit_predict(X)

# #Rekursiv Feature Elimination - selektiere  beste Signale aus
# modelle = modell() #max_leaf_nodes=10, random_state=0
# #Define RFE  - RFE(modelle, 5)
# rfe = RFE(modelle)
# #benutze RFE und selektiere  top (5) (features) Parameters 
# fit = rfe.fit(dframe_zu_kombinieren, y)

# feature_names = list(dframe_zu_kombinieren.columns)

# #Create a dataframe for the results 
# df_RFE_results = []
# for i in range(dframe_zu_kombinieren.shape[1]):
#     df_RFE_results.append(
#         {        #data_ansatz2_2.feature_names[i]
#             'Signal_namen': feature_names[i],
#             'Selectiert':  rfe.support_[i],
#             'RFE_ranking':  rfe.ranking_[i],
#         }
#     )

# df_RFE_results = pd.DataFrame(df_RFE_results)
# df_RFE_results.index.name='Columns'

# #Selektiere nur Relevante Signale fürs Modell
# col_true= list(df_RFE_results['Feature_names'].loc[df_RFE_results['Selected'] == True])
# print('Beste Signale\n', col_true)

# #es wird nun mit den Besten Signale gearbeitet
# beste_signal=dframe_zu_kombinieren[col_true]

# #Daten aufteilen
# #independent
# #x_ansatz2 = data_ansatz2_2.loc[:, data_ansatz2_2.columns != 'result']
# #target dependent Variable
# #y_ansatz2 = data_ansatz2_2[['result']].copy()

# #test_size aus split
# testSize = params['split']

# x_train2, x_test2 , Y_train2, Y_test2 = \
#     train_test_split(dframe_zu_kombinieren,y, test_size=testSize, random_state=325)

# #Die Daten speichern
# x_train2.to_csv("Aufbereiten/train/x_train.csv", index=False)
# Y_train2.to_csv("Aufbereiten/train/y_train.csv", index=False)
# x_test2.to_csv("Aufbereiten/test/x_test.csv",  index=False)
# Y_test2.to_csv("Aufbereiten/test/y_test.csv", index=False)