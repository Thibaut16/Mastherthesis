import io
import os
import os.path
import random
import re
import sys
import xml.etree.ElementTree
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from pandas.io.parsers import read_table
import glob
import yaml
import smtplib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#split zelle knn

# model_2= DecisionTreeClassifier()
# model_2.fit(X_train, y_train)

# print(model_2.score(X_test, y_test))
#read params

params = yaml.safe_load(open("params.yaml"))["extrahieren"]

if len(sys.argv) != 1: #auszufuerende python Datei + daten.csv 
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython 0-extrahieren.py data-file\n")
    sys.exit(1)

#get params
# signal_1, min1, mean1, max1 = params['ges_one'],params['ges_two'],params['ges_three'],\
#                              params['ges_four']
# signal_2, min2, mean2, max2 = params['bsl_one'],params['bsl_two'],params['bsl_three'],\
#                              params['bsl_four']
# signal_3, min3, mean3, max3 = params['kbi_one'],params['kbi_two'],params['kbi_three'],\
#                              params['kbi_four']
# signal_4, min4, mean4, max4 = params['MO_one'],params['MO_two'],params['MO_three'],\
#                              params['MO_four']
#zeit = params['time']

signal_1, min1, mean1, max1 ='KBI_angez_Geschw[Unit_KiloMeterPerHour]',0,100.167423,168
signal_2, min2, mean2, max2 ='BSL_ZAS_Kl_15[]',0,1,1
signal_3, min3, mean3, max3 ='KBI_Kilometerstand[Unit_KiloMeter]',0, 36856.50,37088
signal_4, min4, mean4, max4 ='MO1_Drehzahl_valid[1]',0,0.9,1

#suche = params['suche'] #suche nach params
#folder, um datei zu speichern
data_path = os.path.join("data", "extrahiert")
os.makedirs(data_path, exist_ok=True)

print(len(sys.argv),'', sys.argv)
#fetch data- .csv
#data = sys.argv[0]

#Signale nach float ueberpruefen, Fehler fangen


#selektiere signale aus dem Datensatz
# def select(DF,cols):
#     '''
#     selektiert die Signale nacheinander
#     '''
#     if len(cols)==0:
#       df_ana = DF[["KBI_angez_Geschw[Unit_KiloMeterPerHour]",\
#                         "BSL_ZAS_Kl_15[]", "KBI_Kilometerstand[Unit_KiloMeter]",\
#                         "MO1_Drehzahl_valid[1]" ]]
#       print('ok')
#     else:
#         df_ana =DF [cols]
#     return df_ana

def convert(s):
    '''
    diese hilft die Signale nach Float Datentypen nacheinander zu konvertieren
    '''
    #def ersetzen(s):
    i=str(s).find(',')
    if(i>0):
        return s[:i] + '.' + s[i+1:]
    else :
        return s 


def ueberpruefe_signal(arg1,arg):#*args
    '''
    testet ob fuer das signal die deskriptive Werte akzeptable(signifikant klein) 
    sind und selektiert werden sollen
    '''
    add= False
    for argu in arg:
        param=argu[0]
        test=[p for p in argu if p!=param]

        if arg1 ==test: #best case
            add=True
            return add,param
        else:
            return False , param
        # else:
        #     min,mean,max = arg1[0], arg1[1], arg1[2]
        #     miN, meaN, maX= arg[1], arg[2], arg[3]
        #     res1,res2,res3 = miN-min, meaN-mean,maX-max
        #     if res1<=50 and res2<=50 and res3<=50:
        #        add = True
        #        return True, arg[0]
    return add #, arg[0]

###---Beispiel: text_to_append(signal)='hi,'+str(1)+','+str(10)+','+str(16)--###
def append_neu_zeile(file_name, text_to_append):
    """addiere neue zeile in datei"""
    with open(file_name,"a+") as file_object:
        #read cursor zum Anfang bringen
        file_object.seek(0)
        #falls datei nicht leer addiere \n
        data= file_object.read(100)
        if len(data)>0:
            file_object.write("\n")
        #addiere texte
        file_object.write(text_to_append)

###--Ueberpruefe, ob Parameter (Signal) in datei--##
def param_in_datei(col):
    '''Ueberpruefe, ob Parameter (Signal) in datei'''
    if os.path.exists('feature_store/signal.txt'):
        #falls es existiert, signal hierausholen
        with open('feature_store/signal.txt') as file:
           '''lese alle Signale und Deskriptive Werte'''
           lines = file.readlines()
           lines =[ line.rstrip() for line in lines]
        signalColumns= []
        cols =[]
        sig_gespeichert =[]
        for signalColumn in lines:
            signalCol = signalColumn.split(",")
            cols.append(signalCol[0])
        if col in cols:
            return True
        else:
            return False
    else:
        return False

def extrahiertDaten(dateiname,df):
    '''
    selektiere Parameter aus dem Datensatz
    '''
    col_not_there= True
    #read data funktioniert noch nicht als Eingabe
    #"C:/Users/I008470/Documents/arbeit/masterarbeit_code/jordis-masterthesis/src/export.csv"

    if os.path.exists('feature_store/signal.txt'):
        #falls es existiert, signal hierausholen
        print('Aus Datei Signal entnommen!!!!\n')
        with open('feature_store/signal.txt') as file:
           '''lese alle Signale und Deskriptive Werte'''
           lines = file.readlines()
           lines =[ line.rstrip() for line in lines]
        signalColumns= []
        cols =[]
        sig_gespeichert =[]
        for signalColumn in lines:
            signalCol = signalColumn.split(",")
            cols.append(signalCol[0])
            sig_gespeichert.append(signalCol) #test lernender Prozess
            signalCol.pop(0) #signal schon gespeichert
            signalColumns.append(signalCol)
           
    else:
        #signal aus Stufe 0
        print('Aus Stufe 0 Signal entnommen!!!!\n')

        signal_1, signal_2, signal_3, signal_4='KBI_angez_Geschw[Unit_KiloMeterPerHour]',\
        'BSL_ZAS_Kl_15[]','KBI_Kilometerstand[Unit_KiloMeter]','MO1_Drehzahl_valid[1]'

        print(signal_1,signal_2,signal_3,signal_4,'\n')

        cols=[signal_1, signal_2, signal_3, signal_4,'time']
        sig_gespeichert=[[signal_1,min1,mean1,max1], [signal_2,min2,mean2,max2],\
                        [signal_3,min3,mean3,max3],[signal_4,min4,mean4,max4]]

    col_for_df = [] #selektiert moegliche Signale

    #Datentyp Konvertierung
    print('Datentypen Überprüfen\n')
    for col in df.columns:
      ###Behandle Fehler, sollte eine unerwartete Fehler auftauchen
      try:
          df[col]=df[col].apply(convert)
          df[col] = pd.to_numeric(df[col],errors = 'coerce')
          break
      except:
          print("Hier!", sys.exc_info()[0], "ist aufgetaucht.")
          print("Nächstes Signal")
          print()
    ###--konnte man den Schnitt finden--###
    if set(cols).issubset(list(df.columns)): # durschnitt
        print('Signal mit Name im neuen Datensatz vorhanden..\n', 'Dateiname:',dateiname)
        print("data/extrahiert/"+dateiname)
        print('Ausgewaehlt:','\n',cols,'\n')

        rest=[col for col in df.columns if col not in cols]

        for indx,thisSig in enumerate(rest):
              if not (df.empty):
                min, median,max= df[thisSig].min(),df[thisSig].median(),\
                  df[thisSig].max() 
              else:
                  min, median, max= -1,-1,-1
              print('Nummer:',indx,'','name:',thisSig,'','min:',min,\
                  '','median:',median,'','max:',max,'\n' )
        nutzer_ok=str(input("sind Sie einverstanden oder Weitere Signal? ja oder nein.."))
        if nutzer_ok=='ja':
           print('Nutzer ok--speichern ....')
           data_frame= df[cols] #select(df,[])
           #os.path.join(data_path+"extrahiert"+".csv")
           #data_frame.to_csv("data/extrahiert/export.csv", index=False) #extrahierte Daten
           data_frame.to_csv("data/extrahiert"+dateiname, index=False) #extrahierte Daten
           col_not_there = False
           for s in cols:
               if s!='time':
                 min, median,max= df[s].min(),df[s].median(),df[s].max()
                 signal_neu=s+','+str(min)+','+str(median)+','+str(max)
               else:
                    signal_neu=s
               if param_in_datei(s) == False:
                 append_neu_zeile("feature_store/signal.txt", signal_neu)
           
        if nutzer_ok=='nein':
           auswahl=str(input('selekiere eins oder mehr, Leertaste zwischen Ihrer\
                Eingaben z.B 1 2 3 ->also immer Leertaste dazwischen..'))
           selekt=(auswahl.split(" "))
           selekT=[int(c) for c in selekt]
           for choice in selekT:
                col_for_df.append(rest[choice])
                min, median,max= df[rest[choice]].min(),\
                    df[rest[choice]].median(),df[rest[choice]].max()
                #speichere Auswahl vom Nutzer
                signal_neu=rest[choice]+','+str(min)+','+str(median)+','+str(max)
                if not param_in_datei(rest[choice]):
                  append_neu_zeile("feature_store/signal.txt", signal_neu)  
                #beenden=input("sind Sie zufrieden(fertig) mit der Wahl, geben sie True an \
                #sonst False")
           colsig=[s for s in cols if s not in col_for_df] #weitere Signal betrachten
           for s in colsig:
               min, median,max= df[s].min(),df[s].median(),df[s].max()
               signal_neu=col+','+str(min)+','+str(median)+','+str(max)
               if not param_in_datei(col):
                 append_neu_zeile("feature_store/signal.txt", signal_neu)
           data_frame= df[colsig]
           data_frame.to_csv("data/extrahiert"+dateiname, index=False)
        print('###-speichern fertig--###')
        return

    #Anstatt Deskriptive Werte zu speichern, speichere den Datensatz so
    else: 
        #try to search for signals
        #par1,par2,par3,par4=[signal_1,min1,mean1,max1],[signal_2,min2,mean2,max2],\
           # [signal_3,min3,mean3,max3],[signal_4,min4,mean4,max4]
        print('Es werden nach Signale anhand von Deskriptiven Werte min, max,\
        median gesucht....\n')
        for col in df.columns:
            min,mean,max=df[col].min(), df[col].median(), df[col].max()
            #add,signal = ueberpruefe_signal([min,mean,max],par1,par2,par3,par4)
            #man bildet die Schnittmenge
            add,signal = ueberpruefe_signal([min,mean,max],sig_gespeichert)

            if add: #Signal hat genau gleiche verteilung
                col_for_df.append(col) #spalte wird aufgenommen
                #ist "KBI_angez_Geschw[Unit_KiloMeterPerHour]" gefunden worden?
                #if signal=="KBI_angez_Geschw[Unit_KiloMeterPerHour]":
                   #merke Name, fuer spaeteren Aufruf
                #    f = open("signal.txt", "a")
                #    f.write(col)
                #    f.close()
                #--man merkt Signale die, anders heißen, aber gleich sind--#
                signal_neuName=col+','+signal
                #speichere gleicher Name
                append_neu_zeile("feature_store/signal-name.txt", signal_neuName)
                #speichere mit den anderen Signale
                signal_neu=col+','+str(min)+','+str(mean)+','+str(max)
                if not param_in_datei(col):
                   append_neu_zeile("feature_store/signal.txt", signal_neu)


    # if len(col_for_df)!=0:
    #         DF = df[col_for_df]  #select(df, col_for_df)
    #         #(os.path.join(data_path,"extrahiert"+".csv")) 
    #         DF.to_csv("data/extrahiert/export.csv", index=False)#extrahierte Daten
    #         return

    #if  col_not_there:
        #auch wenn Signal nicht da ist weiter machen
    print('ausgewälht::',col_for_df,'\n')
    #gebliebende Signale werden dem Nutzer vorgeschlagen, dann entscheidet er
    if len(col_for_df)!=0:
      print('restlichen Parameters werden selektiert zur Vorwahl\n')
      gebliebene_sig = [col for col in df.columns if col not in col_for_df]
    else:
       gebliebene_sig = [col for col in df.columns]
    print(len(gebliebene_sig))
    if len(gebliebene_sig)!=0:
        print('\n','col_df_old:',list(df.columns),'\n')
        print('\n','col_geblieben:',gebliebene_sig,len(gebliebene_sig),'\n','\n',\
              'lange unterschied:',len(list(df.columns))-len(gebliebene_sig))
        print('\n')
        for indx,thisSig in enumerate(gebliebene_sig):
              if not (df.empty):
                min, median,max= df[thisSig].min(),df[thisSig].median(),\
                  df[thisSig].max() 
              else:
                  min, median, max=-1,-1,-1
              print('Nummer:',indx,'','name:',thisSig,'','min:',min,\
                  '','median:',median,'','max:',max,'\n' )
        auswahl=str(input("selekiere eins oder mehr, Leertaste zwischen Ihrer\
                Eingaben z.B 1 2 3 ->also immer Leertaste dazwischen.."))
        selekt=(auswahl.split(" "))
        selekT=[int(c) for c in selekt]
        for choice in selekT:
           col_for_df.append(gebliebene_sig[choice])
           min, median,max= df[gebliebene_sig[choice]].min(),\
               df[gebliebene_sig[choice]].median(),df[gebliebene_sig[choice]].max()
           #speichere Auswahl vom Nutzer
           signal_neu=gebliebene_sig[choice]+','+str(min)+','+str(mean)+','+str(max)
           if not param_in_datei(gebliebene_sig[choice]):
             append_neu_zeile("feature_store/signal.txt", signal_neu)  

    df=df[col_for_df]
    df.to_csv("data/extrahiert/"+dateiname, index=False)
    #print("Signale nicht gefunden")
        #exit(0)
    return
        # try:
        # except:

#if not suche:

path = r'C:/Users/I008470/Documents/arbeit/masterarbeit_code/jordis-masterthesis/evaluate/'
all_daten= glob.glob(path+"/*.csv")
li=[]
for dateiname in all_daten:
    df = pd.read_csv(dateiname, sep=",")
    #df = pd.read_csv(dateiname, index_col=None, header=0)
    #df = pd.read_csv("src/export.csv" , sep=";") #sep=","
    dateiname=dateiname[-11:]
    extrahiertDaten(dateiname, df)

# else:
#     df = pd.read_csv("src/export.csv" , sep=";") #sep=","
#     df.to_csv("data/extrahiert/export.csv", index=False)