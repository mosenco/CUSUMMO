# packages
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw,similarity
from scipy.stats import pearsonr
from sklearn.model_selection import ParameterGrid
import statistics

from claspy.segmentation import BinaryClaSPSegmentation

import ruptures as rpt

import stumpy
from aeon.segmentation import find_dominant_window_sizes

from aeon.segmentation import GreedyGaussianSegmenter

from aeon.segmentation import InformationGainSegmenter

from aeon.anomaly_detection import STRAY

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer,mean_squared_error
from ruptures.metrics import precision_recall
import matplotlib.pyplot as plt
#from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import seaborn as sns

sns.set_theme()
sns.set_color_codes()

from claspy.tests.evaluation import f_measure,covering

from claspy.window_size import dominant_fourier_frequency, highest_autocorrelation, suss

"""
Extracted features. Use the index of this list to use with iloc[]

0. Kinetic Global
1. Kinetic Chest
2. Directness Head
3. Density
4. left wrist ke
5. right wrist ke
6. left ankle ke
7. right ankle ke
8. head ke
9. crouch density
10. left leg density
11. right leg density
12. left hand density
13. right hand density
14. head density
15. arto inferiore
16. gamba
17. coscia
18. coscia dx
19. coscia sx
20. gamba sx
21. gamba dx
22. braccio sx
23. braccio dx
24. avambraccio sx
25. avambraccio dx
26. ARIEL speed magnitude
27. ARIEL speed X component
28. ARIEL speed Y component
29. ARIEL speed Z component
30. ARIEL acceleration magnitude
31. ARIEL acceleration X component
32. ARIEL acceleration Y component
33. ARIEL acceleration Z component
34. ARIEL jerk magnitude
35. ARIEL jerk X component
36. ARIEL jerk Y component
37. ARIEL jerk Z component
38. STRN speed magnitude
39. STRN speed X component
40. STRN speed Y component
41. STRN speed Z component
42. STRN acceleration magnitude
43. STRN acceleration X component
44. STRN acceleration Y component
45. STRN accelerationZ component
46. STRN jerk magnitude
47. STRN jerk X component
48. STRN jerk Y component
49. STRN jerk Z component
50. RHEL speed magnitude
51. RHEL speed X component
52. RHEL speed Y component
53. RHEL speed Z component
54. RHEL acceleration magnitude
55. RHEL acceleration X component
56. RHEL acceleration Y component
57. RHEL acceleration Z component
58. RHEL jerk magnitude
59. RHEL jerk X component
60. RHEL jerk Y component
61. RHEL jerk Z component
62. LHEL speed magnitude
63. LHEL speed X component
64. LHEL speed Y component
65. LHEL speed Z component
66. LHEL acceleration magnitude
67. LHEL acceleration X component
68. LHEL acceleration Y component
69. LHEL acceleration Z component
70. LHEL jerk magnitude
71. LHEL jerk X component
72. LHEL jerk Y component
73. LHEL jerk Z component
74. RPLM speed magnitude
75. RPLM speed X component
76. RPLM speed Y component
77. RPLM speed Z component
78. RPLM acceleration magnitude
79. RPLM acceleration X component
80. RPLM acceleration Y component
81. RPLM acceleration Z component
82. RPLM jerk magnitude
83. RPLM jerk X component
84. RPLM jerk Y component
85. RPLM jerk Z component
86. LPLM speed magnitude
87. LPLM speed X component
88. LPLM speed Y component
89. LPLM speed Z component
90. LPLM acceleration magnitude
91. LPLM acceleration X component
92. LPLM acceleration Y component
93. LPLM acceleration Z component
94. LPLM jerk magnitude
95. LPLM jerk X component
96. LPLM jerk Y component
97. LPLM jerk Z component





"""
# list of features. To access its name or its value while using iloc
features_name=[
    "kinetic_global",
    "kinetic_chest",
    "directness_head",
    "density",
    "left_wrist_ke",
    "right_wrist_ke",
    "left_ankle_ke",
    "right_ankle_ke",
    "head_ke",
    "crouch_density",
    "left_leg_density",
    "right_leg_density",
    "left_hand_density",
    "right_hand_density",
    "head_density",
    "arto_inferiore",
    "gamba",
    "coscia",
    "coscia_dx",
    "coscia_sx",
    "gamba_sx",
    "gamba_dx",
    "braccio_sx",
    "braccio_dx",
    "avambraccio_sx",
    "avambraccio_dx",
    "ARIEL_speed_magnitude",
    "ARIEL_speed_X_component",
    "ARIEL_speed_Y_component",
    "ARIEL_speed_Z_component",
    "ARIEL_acceleration_magnitude",
    "ARIEL_acceleration_X_component",
    "ARIEL_acceleration_Y_component",
    "ARIEL_acceleration_Z_component",
    "ARIEL_jerk_magnitude",
    "ARIEL_jerk_X_component",
    "ARIEL_jerk_Y_component",
    "ARIEL_jerk_Z_component",
    "STRN_speed_magnitude",
    "STRN_speed_X_component",
    "STRN_speed_Y_component",
    "STRN_speed_Z_component",
    "STRN_acceleration_magnitude",
    "STRN_acceleration_X_component",
    "STRN_acceleration_Y_component",
    "STRN_acceleration_Z_component",
    "STRN_jerk_magnitude",
    "STRN_jerk_X_component",
    "STRN_jerk_Y_component",
    "STRN_jerk_Z_component",
    "RHEL_speed_magnitude",
    "RHEL_speed_X_component",
    "RHEL_speed_Y_component",
    "RHEL_speed_Z_component",
    "RHEL_acceleration_magnitude",
    "RHEL_acceleration_X_component",
    "RHEL_acceleration_Y_component",
    "RHEL_acceleration_Z_component",
    "RHEL_jerk_magnitude",
    "RHEL_jerk_X_component",
    "RHEL_jerk_Y_component",
    "RHEL_jerk_Z_component",
    "LHEL_speed_magnitude",
    "LHEL_speed_X_component",
    "LHEL_speed_Y_component",
    "LHEL_speed_Z_component",
    "LHEL_acceleration_magnitude",
    "LHEL_acceleration_X_component",
    "LHEL_acceleration_Y_component",
    "LHEL_acceleration_Z_component",
    "LHEL_jerk_magnitude",
    "LHEL_jerk_X_component",
    "LHEL_jerk_Y_component",
    "LHEL_jerk_Z_component",
    "RPLM_speed_magnitude",
    "RPLM_speed_X_component",
    "RPLM_speed_Y_component",
    "RPLM_speed_Z_component",
    "RPLM_acceleration_magnitude",
    "RPLM_acceleration_X_component",
    "RPLM_acceleration_Y_component",
    "RPLM_acceleration_Z_component",
    "RPLM_jerk_magnitude",
    "RPLM_jerk_X_component",
    "RPLM_jerk_Y_component",
    "RPLM_jerk_Z_component",
    "LPLM_speed_magnitude",
    "LPLM_speed_X_component",
    "LPLM_speed_Y_component",
    "LPLM_speed_Z_component",
    "LPLM_acceleration_magnitude",
    "LPLM_acceleration_X_component",
    "LPLM_acceleration_Y_component",
    "LPLM_acceleration_Z_component",
    "LPLM_jerk_magnitude",
    "LPLM_jerk_X_component",
    "LPLM_jerk_Y_component",
    "LPLM_jerk_Z_component",
]

TIMESERIES=[
    "in\cora1_in.txt",
      "in\cora4_05_in.txt",
      "in\cora4_08_in.txt",
      "in\cora5_in.txt",
      "in\cora14_in.txt",
      "in\marianne7_in.txt",
      "in\marianne8_in.txt",
      "in\marianne10_in.txt",
      "in\marianne18_in.txt",
      "in\marianne19_in.txt",
      "in\marianne24_in.txt",
      "in\marianne26_in.txt",
      "in\marianne41_in.txt",
      "in\marianne42_in.txt",
      "in\marianne43_in.txt",
      "in\marianne47_in.txt",
      "in\marianne48_in.txt",
      "in\muriel18_in.txt",
      "in\muriel26_in.txt",
      "in\muriel27_in.txt",
      "in\muriel30_in.txt"

      ]
GROUNDTRUTH=[
         "gt\cora_gt_2019-08-08_t001_video01.txt",
         "gt\cora_gt_2019-05-22_t004_video01.txt",
         "gt\cora_gt_2019-08-08_t004_video01.txt",
         "gt\cora5_gt.txt",
         "gt\cora_gt_2019-08-08_t014_video01.txt",
         "gt\marianne_gt_2016-03-22_t007_video01.txt",
         "gt\marianne_gt_2016-03-22_t008_video01.txt",
         "gt\marianne_gt_2016-03-22_t010_video01.txt",
         "gt\marianne_gt_2016-03-22_t018_video01.txt",
         "gt\marianne_gt_2016-03-22_t019_video01.txt",
         "gt\marianne_gt_2016-03-22_t024_video01.txt",
         "gt\marianne_gt_2016-03-22_t026_video01.txt",
         "gt\marianne_gt_2016-03-22_t041_video01.txt",
         "gt\marianne_gt_2016-03-22_t042_video01.txt",
         "gt\marianne_gt_2016-03-22_t043_video01.txt",
         "gt\marianne_gt_2016-03-22_t047_video01.txt",
         "gt\marianne_gt_2016-03-22_t048_video01.txt",
         "gt\muriel_gt_2016-03-21_t018_video01.txt",
         "gt\muriel_gt_2016-03-21_t026_video01.txt",
         "gt\muriel_gt_2016-03-21_t027_video01.txt",
         "gt\muriel_gt_2016-03-23_t030_video01.txt"
         ]



def f1scoremargin(ground_truth, predictions, tolerance):
    """
    Calcola l'F1 score con una finestra di tolleranza sui change points.
    
    :param ground_truth: Lista o array di change points reali
    :param predictions: Lista o array di change points predetti
    :param tolerance: La tolleranza temporale (numero di unità temporali)
    :return: precision, recall, f1-score
    """
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    # list of correct prediction
    correct_pred= []
    # list of false positives
    false_pred=[]
    # Vettori per tracciare quali punti sono stati già associati
    matched_ground_truth = np.zeros(len(ground_truth), dtype=bool)
    matched_predictions = np.zeros(len(predictions), dtype=bool)

    mgt={key: False for key in ground_truth}
    mcp={key: False for key in predictions}
    #print(f'gt:{len(ground_truth)} - cp:{len(predictions)}')
    # True Positives (TP)
    tp = 0
    closer=predictions[0]
    c_idx=0
    for i, gt_point in enumerate(ground_truth):
        # errato. bisogna guardare il punto piu vicino
        """
        for j, pred_point in enumerate(predictions):
            
            if not matched_predictions[j] and abs(gt_point - pred_point) <= tolerance:
                tp += 1
                matched_ground_truth[i] = True
                matched_predictions[j] = True

                mgt[gt_point] = True
                mcp[pred_point] = True
                break
        """
        # find the closest
        #closer=predictions[0]
        #for j, pred_point in enumerate(predictions):
        for j in range(c_idx,len(predictions)):
            pred_point = predictions[j]
            # if im over the true cp, no reason to check for next
            # because the distance will just increase
            if pred_point > gt_point and abs(gt_point - pred_point) > tolerance:
                break
            if abs(gt_point - pred_point) < abs(gt_point - closer):
                closer = pred_point
                c_idx = j
        # reached this point we get a closer prediction and we check if it's inside the window
        if not matched_predictions[c_idx] and abs(gt_point - closer) <= tolerance:
            tp += 1
            matched_ground_truth[i] = True
            matched_predictions[c_idx] = True
            correct_pred.append(closer)

    # after completed corrected, we need to create our false positive list
    for i,pred in enumerate(predictions):
        if not matched_predictions[i]:
            false_pred.append(pred)
    
    # False Positives (FP) - predizioni non corrispondenti a nessun ground truth entro la tolleranza
    fp = np.sum(~matched_predictions)
    
    # False Negatives (FN) - punti del ground truth non corrispondenti a nessuna predizione entro la tolleranza
    fn = np.sum(~matched_ground_truth)
    #print(f'tp:{tp} - fp:{fp} - fn:{fn}')
    #print(mgt)
    #print(mcp)
    # Calcolo di precision, recall e F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #print(f'gt:{len(ground_truth)} cp:{len(predictions)} tp:{tp} fp:{fp} fn:{fn}')
    return precision, recall, f1, {"tp":tp, "fp":fp, "fn":fn}, correct_pred, false_pred




def ReadAndPreProcess(inputDataRaw):
    # lettura
    df=pd.read_csv(inputDataRaw,sep=' ', header=None).interpolate()
    
    df=df.drop(0, axis=1)
    df=df.drop_duplicates()
    df = df.iloc[:, ::-1]
 

    return df



# questa funzione ritorna un dataframe del groundtruth che viene usato specificatamente per visualizzare il gt
# è soggetto a un preprocessing dei dati siccome l'ultimo groundtruth è dove termina il ts del gt
# di conseguenza per farlo corrispondere, bisogna stretcharlo
# ma ricordo di aver rifatti i dati nuovi per generare un groundtruth a fine ts, da controllare cosi che non serve stretcharlo?
def LoadingGroundTruth(df,gtraw):
    gt=pd.read_csv(gtraw,sep=' ', header=None)
    gt=gt.iloc[:,0].values
    #stretching dei dati se necessario per farlo corrispondere alla ts dei dati
    stretch_gt = np.array([])
    for idx,i in enumerate(gt):
        relpos = len(df)*i/gt[-1]
        stretch_gt = np.append(stretch_gt,relpos)

    # eliminiamo l'ultimo elemento che è stato annotato solo per delimitare la lunghezza della gt simile alla ts
    
    return stretch_gt[:-1]


def GetClasp2(df,gt,known,feature, **kwargs):
    
    result=np.array([])
    eachresult = []
    eachclasp=[]
    for i in feature:
    
        ts=df.iloc[:,i]
        
        #print(ts.head())
        if known == 1:
            #print("knwon!")
            clasp = BinaryClaSPSegmentation(n_segments=len(gt), validation=None)
        else:
            #print("unknown!")
            clasp = BinaryClaSPSegmentation(**kwargs)
            
        found_cps = clasp.fit_predict(ts.values)    

        # c'è un bug con binseg dove un cp è oltre la lunghezza del ts
        # faccio un loop e se eccede cambio il valore con la len(tf)-1
        # WTF IS THIS
        """
        for i in range(0,len(found_cps)):
            if found_cps[i] >= len(ts):
                found_cps[i] = len(ts)-1
        """
        # per ogni array di cp di ogni singola feature
        # li unisco in un unico array. in pratica faccio un OR di tutti i cp
        result = np.sort(np.append(result,found_cps).flatten())
        #potenziale bug.
        #se faccio unique() mi toglie il numero di cp in un punto e quando faccio majority voting mi si toglie
        result = np.unique(result)
        eachresult.append(found_cps)
        eachclasp.append(clasp)
        
        

        
    return result, eachresult, eachclasp



# utilizzo CLASP 
# prende come parametro un dataframe e restituisce il clasp score
# gt e known vengono usati per usare il numero vero di cp se uguale a 1 sennò si cerca di predirlo se il modello lo permette
def GetClasp3(df,gt,known,feature, **kwargs):
    
    result=np.array([])
    eachresult = []
    eachclasp=[]
    for i in [feature]:
    
        ts=df.iloc[:,i]
        
        #print(ts.head())
        if known == 1:
            #print("knwon!")
            clasp = BinaryClaSPSegmentation(n_segments=len(gt), validation=None)
        else:
            #print("unknown!")
            clasp = BinaryClaSPSegmentation(**kwargs)
            
        found_cps = clasp.fit_predict(ts.values)    

        # c'è un bug con binseg dove un cp è oltre la lunghezza del ts
        # faccio un loop e se eccede cambio il valore con la len(tf)-1
        # WTF IS THIS
        """
        for i in range(0,len(found_cps)):
            if found_cps[i] >= len(ts):
                found_cps[i] = len(ts)-1
        """

        # per ogni array di cp di ogni singola feature
        # li unisco in un unico array. in pratica faccio un OR di tutti i cp
        result = np.sort(np.append(result,found_cps).flatten())
        result = np.unique(result)
        eachresult.append(found_cps)
        eachclasp.append(clasp)
        
        

        
    return result, eachresult, eachclasp



# utilizzo CLASP 
# prende come parametro un dataframe e restituisce il clasp score
# gt e known vengono usati per usare il numero vero di cp se uguale a 1 sennò si cerca di predirlo se il modello lo permette
def GetClasp(df,gt,known, **kwargs):
    
    result=np.array([])
    eachresult = []
    eachclasp=[]
    for i in range(0,len(features_name)):
    
        ts=df.iloc[:,i]
        
        #print(ts.head())
        if known == 1:
            #print("knwon!")
            clasp = BinaryClaSPSegmentation(n_segments=len(gt), validation=None)
        else:
            #print("unknown!")
            clasp = BinaryClaSPSegmentation(**kwargs)
            
        found_cps = clasp.fit_predict(ts.values)    

        # c'è un bug con binseg dove un cp è oltre la lunghezza del ts
        # faccio un loop e se eccede cambio il valore con la len(tf)-1
        # WTF IS THIS
        """
        for i in range(0,len(found_cps)):
            if found_cps[i] >= len(ts):
                found_cps[i] = len(ts)-1
        """
        # per ogni array di cp di ogni singola feature
        # li unisco in un unico array. in pratica faccio un OR di tutti i cp
        result = np.sort(np.append(result,found_cps).flatten())
        result = np.unique(result)
        eachresult.append(found_cps)
        eachclasp.append(clasp)
        
        

        
    return result, eachresult, eachclasp
        


def PlotResult(df,gt,cp, nomeFile, margin,clasplist,ts):
    #da testare quando ho piu valori
    #clasp.plot(gt_cps=gt.astype(int), heading="Segmentation of different umpire cricket signals", ts_name="ACC", file_path="segmentation_example.png")
    print("asd")
    # from f1scoremargin i extract the score,TP and FP
    precision,recall,f1,score,TP,FP=f1scoremargin(gt.astype(int),np.array(cp).astype(int),margin)
    
    if nomeFile == "rplm":
        excl=[0,1,2,3,5,6,7,8,9,12,13,14,15,16]
    elif nomeFile == "ariel":
        excl=[8,9,12]
    elif nomeFile == "lplm":
        excl=[0,1,4,5,6,8,9,10,11,12,15,16]
    elif nomeFile == "rhel":
        excl=[16,17]
    elif nomeFile == "strn":
        excl=[8,9,12,16,17]
    else:
        excl=[]
    plt.figure(figsize=(18,9))
    plt.plot(np.arange(len(df.iloc[:,ts].values)),df.iloc[:,ts].values,'black',linewidth=0.5)
    #for j in cp.tolist():
    for j in FP:
        plt.axvline(x = j, color = 'black',linewidth=1,linestyle="-.",alpha=0.5)
    for j in TP:
        
        plt.axvline(x = j, color = 'black',linewidth=1,linestyle="-.",alpha=0.5) 
    for idx,i in enumerate(gt.astype(int)):
    
        if idx in excl:
            plt.axvline(x = i, color = 'red',linewidth=1) 
        else:
            plt.axvline(x = i, color = 'red',linewidth=1,linestyle="-.",alpha=0.5) 
            
    

    for k in gt.astype(int):
        pass
        #plt.fill_betweenx(np.array([0, 1]), k-margin, k+margin, color='green', alpha=0.3)
    plt.xlabel(f'{nomeFile} {clasplist} {f1}')

    #plt.figure(figsize=(18,9))

 
def PlotResultColored(df,gt,cp, colored, nomeFile, margin,clasplist,ts):
    #da testare quando ho piu valori
    #clasp.plot(gt_cps=gt.astype(int), heading="Segmentation of different umpire cricket signals", ts_name="ACC", file_path="segmentation_example.png")
    
    plt.figure(figsize=(18,9))
    plt.plot(np.arange(len(df.iloc[:,ts].values)),df.iloc[:,ts].values,'blue',linewidth=0.5)
    for idx,j in enumerate(cp.tolist()):
        if idx in colored:
            plt.axvline(x = j, color = 'red',linewidth=2) 
        else:
            plt.axvline(x = j, color = 'green',linewidth=2) 
    for idx,i in enumerate(gt.astype(int)):
        
        plt.axvline(x = i, color = 'blue',linewidth=1,linestyle="-.",alpha=1) 
            
    

    for k in gt.astype(int):
        plt.fill_betweenx(np.array([0, 1]), k-margin, k+margin, color='green', alpha=0.3)
    plt.xlabel(f'{nomeFile} {clasplist} {f1scoremargin(gt.astype(int),cp.astype(int),margin)}')

    #plt.figure(figsize=(18,9))


def Plotclasp(eachclasp,gt,margin,eachcp,feature_list):
    #print("idx"+str(asd))
    
    for idx,clasp in enumerate(eachclasp):
        print(features_name[feature_list[idx]])
        clasp.plot(gt_cps=gt.astype(int), heading=f'f1margin: {f1scoremargin(gt.astype(int),eachcp[idx].astype(int),margin)}')
        #clasp.plot(gt_cps=None, heading=f'f1margin: {f1scoremargin(gt.astype(int),eachcp[idx].astype(int),margin)}', ts_name="suss")


        plt.xlabel(features_name[feature_list[idx]])
        for idx2,j in enumerate(gt.astype(int)):
            plt.fill_betweenx(np.array([0, 1]), j-margin, j+margin, color='green', alpha=0.3)

        
# calcola i vari scores dati il groundtruth e il prediction
# puo salvare il risultato su file per evitare di perderli
# prende come parametro nome del groundtruth, groundtruth, nome della timeseries e il prediction
def Evaluate(modelName,gtName, gt, tsName, cp, df, margin):
    # creo dei array di lunghezza come la ts cosi possono fare il confronto
    # sia per il gt che per il pd
  
    cpnump = np.array(cp)
    gtnump = np.array(gt)

    cp_long = np.zeros(len(df)+1)
    cp_long[cpnump.astype(int)]=1

    gt_long = np.zeros(len(df)+1)
    gt_long[gtnump.astype(int)]=1

    # calcolo lo score 
    print(f'f1margin: {f1scoremargin(gt.astype(int),cp.astype(int),margin)}')
    return f1scoremargin(gt.astype(int),cp.astype(int),margin)
    


def IgnoreZone(idx,cpraw,gt,margin):
    cp = cpraw.tolist()
    if idx == 0: #cora1
        for i in range(len(cp)-1, -1, -1):
            #if cp[i] > 3944.7118557910376+100 and cp[i] < 5911.693516853054-100 or cp[i] > 12845.0+100:
            if cp[i] > gt[18]+margin and cp[i] < gt[19]-margin or cp[i]> gt[-1]+margin:
                cp.pop(i)
                
        
    elif idx == 1: #cora4_05
        for i in range(len(cp)-1, -1, -1):
            #if cp[i]< 969.6180827886711-100 and cp[i] > 13125.469063180826+100:
            if cp[i] < gt[0]-margin or cp[i] > gt[-1]+margin:
                cp.pop(i)
    elif idx == 2: #cora4_08
        for i in range(len(cp)-1, -1, -1):
            #if cp[i] > 2874.607407407407+100 and cp[i] < 4016.935849056604-100:
            if cp[i] > gt[-2]+margin and cp[i] < gt[-1]-margin or cp[i] < gt[0]-margin:
                cp.pop(i)

    elif idx == 4: #cora14
        for i in range(len(cp)-1,-1,-1):
            if cp[i] < gt[0] - margin:
                cp.pop(i)

    elif idx == 5: #marianne7
        for i in range(len(cp)-1, -1, -1):
            if cp[i] < gt[0] - margin:
                cp.pop(i)

    elif idx == 17: #muriel18
        for i in range(len(cp)-1, -1, -1):
            #if cp[i] > 180.03455207940698+100 and cp[i] < 1227.051137077522-100 or cp[i] > 5865.505591154668+100:
            if cp[i] > gt[0]+margin and cp[i] < gt[1]-margin or cp[i] > gt[-1]+margin:
                cp.pop(i)

    elif idx == 18: #muriel26
        for i in range(len(cp)-1, -1, -1):
            #if cp[i] > 138.33224102186853+100 and cp[i] < 3677.231833076974-100:
            if cp[i] > gt[0]+margin and cp[i] < gt[1]-margin:
                cp.pop(i)


    elif idx == 21: #muriel30
        for i in range(len(cp)-1, -1, -1):
            #if cp[i] > 8187.634803581529+100:
            if cp[i] > gt[26]+margin and cp[i] < gt[27]-margin or cp[i] > gt[-1]+margin:
                cp.pop(i)


    else:
        pass
        #print("error IgnoreZone()")
    return np.array(cp)



def delnear(arr,range):
    i = 0
    while i < len(arr) - 1:
        # Iniziamo con il primo elemento di un potenziale gruppo
        gruppo_inizio = i
        gruppo_fine = i

        # Cerca gli elementi che fanno parte dello stesso gruppo
        while gruppo_fine < len(arr) - 1 and arr[gruppo_fine + 1] - arr[gruppo_fine] < range:
            gruppo_fine += 1

        # Se esiste un gruppo di più elementi
        if gruppo_fine > gruppo_inizio:
            # Se la distanza tra l'inizio e la fine è minore di 50, elimina l'elemento maggiore (gruppo_fine)
            if arr[gruppo_fine] - arr[gruppo_inizio] < range:
                arr = np.delete(arr, gruppo_fine)
            
            # Elimina tutti gli elementi interni al gruppo
            arr = np.concatenate((arr[:gruppo_inizio + 1], arr[gruppo_fine:]))

        # Procedi con il prossimo gruppo
        i = gruppo_inizio + 1

    return arr




def MajorityVoteCP(arr,margin,amount):
    if len(arr)==0:
        return np.array([])
    # se vicino continua ad aggiungere

    # se lontano e bucket presente, generare medio

    # se lontano e bucket vuoto aggiornare start
    bucket=[]
    answer=[]

    for i in range(len(arr)-1,-1,-1):
        if bucket == []:
            bucket.append(arr[i])
        elif abs(arr[i]-bucket[-1]) <= margin:
            bucket.append(arr[i])
        elif abs(arr[i]-bucket[-1]) > margin:
            if len(bucket) < amount:
                bucket=[arr[i]]
            else:
                summ=0
                for j in bucket:
                    summ+=j
                answer.append(summ/len(bucket))
                bucket=[arr[i]]
    if len(bucket) < amount:
        bucket=[]
    else:
        summ=0
        for j in bucket:
            summ+=j
        answer.append(summ/len(bucket))
        bucket=[]
    return np.array(answer)


def FPremoverDTW(featureTS,res,threshold):
     # controlla quale segmento piu lungo o corto
    # si fa una finestra grande quanto corto e slitta su lungo
    # si fa un dtw per ogni finestra e si prende lo score piu basso
    # se entro un certo limite si unisce, senno si continua
    if len(res) < 3:
        return res
    to_color=[]
    filtered=[res[0]]
    lo = 0
    hi = 1
    while hi < len(res)-1:
       # print(f'hi:{hi} max:{len(res)}')
        a = featureTS[int(res[lo]):int(res[hi])]
        b = featureTS[int(res[hi]):int(res[hi+1])]
        # controllo quale segmento piu lungo o corto
        # halflo = segmento piu corto è una timeseries
        # halfhi = segmento piu lungo è una timeseries
        if len(a) < len(b):
            halflo=featureTS[int(res[lo]):int(res[hi])]
            halfhi=featureTS[int(res[hi]):int(res[hi+1])]
        else:
            halfhi=featureTS[int(res[lo]):int(res[hi])]
            halflo=featureTS[int(res[hi]):int(res[hi+1])]

        # faccio sliding window e trovo la distance piu piccola
        st = 0
        en = len(halflo) # possibile bug
        smallestdist=-1
        smalllo=[]
        highlo=[]
      
       # print(f'en:{en} whilelim:{len(halfhi)}')
        while en < len(halfhi):
            
            slidewin = halfhi[st:en]
            halflo_d = np.array(halflo, dtype=np.double)
            slidewin_d = np.array(slidewin, dtype=np.double)
            distance = dtw.distance_fast(halflo_d, slidewin_d, use_pruning=True)
            if smallestdist == -1:
                smallestdist = distance
                smalllo=halflo
                highlo=slidewin
            elif smallestdist > distance:
                smallestdist = distance
                smalllo=halflo
                highlo=slidewin

            st+=1
            en+=1
            #print(f'en:{en} lim:{len(halfhi)} dist:{distance}')
        
        #plt.figure(figsize=(18,9))
        #plt.plot(np.arange(len(smalllo)),smalllo,"blue")
       # plt.figure(figsize=(18,9))
       # plt.plot(np.arange(len(highlo)),highlo,"green")
        #plt.xlabel(f'smallest:{smallestdist}')
        if smallestdist <= threshold:
            #print("removed")
            #print(f'smallest:{smallestdist}')
            to_color.append(hi)
            hi+=1
        else:
            filtered.append(res[hi])
            lo=hi
            hi=lo+1
    filtered.append(res[-1])
    return filtered


def LoadData(path):
    """
            #suss
        with open("../dfl.pkl", "rb") as f:
            dfl = pickle.load(f)

        with open("../gtl.pkl", "rb") as f:
            gtl = pickle.load(f)

        with open("../cpsl.pkl", "rb") as f:
            cpsl = pickle.load(f)

        with open("../cpsl_suss_DTW.pkl", "rb") as f:
            cpsl_dtw = pickle.load(f)

        #normalized
        with open("../dfl_no.pkl", "rb") as f:
            dfl_no = pickle.load(f)

        with open("../cpsl_no.pkl", "rb") as f:
            cpsl_no = pickle.load(f)

        with open("../cpsl_no_DTW.pkl", "rb") as f:
            cpsl_no_dtw = pickle.load(f)

        with open("../cpsl_no_1.pkl", "rb") as f:
            cpsl_no_1 = pickle.load(f)

        #standardized
        with open("../dfl_st.pkl", "rb") as f:
            dfl_st = pickle.load(f)

        with open("../cpsl_st.pkl", "rb") as f:
            cpsl_st = pickle.load(f)

        with open("../cpsl_st_DTW.pkl", "rb") as f:
            cpsl_st_dtw = pickle.load(f)

        with open("../cpsl_st_1.pkl", "rb") as f: #è un dtw... però ancora in testing
            cpsl_st_1 = pickle.load(f)
    
    """
    with open(path,"rb") as f:
        return pickle.load(f)
    


# Dato una serie di cp, li combina in uno e fa il majority voting
def AndSal(margin,majvote,*args):
    result=np.array([])
 
    for cp in args:
        result = np.append(result,cp).flatten()
    result = np.sort(result)
    result = MajorityVoteCP(result,margin,majvote)
    return result


# combina le 3 componenti x,y,z insieme e poi fa un majority voting con tutte le altre feature in lista
def ComputeSaliency(cps,val,*vallist):
    vel=[cps[val[1]],cps[val[2]],cps[val[3]]]
    cp1 = AndSal(100,1,*vel)
    #cp1 = AndSal(100,1,cp,cps[val[0]])

    acc_c=[cps[val[5]],cps[val[6]],cps[val[7]]]
    cp2 = AndSal(100,1,*acc_c)
    #cp2 = AndSal(100,1,cp,cps[val[4]])

    jerk_c=[cps[val[9]],cps[val[10]],cps[val[11]]]
    cp3 = AndSal(100,1,*jerk_c)
    #cp3 = AndSal(100,1,cp,cps[val[8]])
    tot=[cp1,cp2,cp3,cps[val[0]],cps[val[4]],cps[val[8]]]#,cps[val[12]],cps[val[13]],cps[val[14]],cps[val[15]]]
    for i in vallist:
        tot.append(cps[i])
    result = AndSal(100,2,*tot)
   # print(f'mlor:{math.floor(len(tot)/2)+1}')
    #PlotResult(df,gt,result,name,100,"")
    return result


def UnionCP(cps,*val):
    res=np.array([])
    for i in val:
        res = np.append(res,cps[i]).flatten()
    return np.sort(res)


def UnionCPS(*val):
    res=np.array([])
    for i in val:
        res = np.append(res,i).flatten()
    return np.sort(res)




"""
# esperimento dove prima combino i componenti x,y,z con il loro modulo
# poi essi vengono presi e combinati con le feature rimanenti

# DEFAULT DATASET -> XYZCOMBINE -> RESULT
def XYZCombine(cpsl,dtwfilter):
    delnear_am=[]
    majority_am=[]
    cleaned_array = [s[3:-7] for s in timeseries]
    resexcel=pd.DataFrame()
    resexcel["name"] = cleaned_array
    am=0
    delnear_score=[]
    majority_score=[]

        

    
    for k in range(1,2):#5):
        for h in range(1,2):#4):
        
            neg=0
            f1_list=[]
            for i in range(len(timeseries)):
                
                # prendo i dati per il singolo video
                df=dfl[i]
                gt=gtl[i]
                cps=np.array(cpsl[i],dtype="object")

                vel=cps[[26,27,28,29]]
                acc=cps[[30,31,32,33]]
                jerk=cps[[34,35,36,37]]
                ariel = AndSal(100,h,*AndSal(100,k,*vel),*AndSal(100,k,*acc),*AndSal(100,k,*jerk),UnionCP(cps,2,8,14))

                vel=cps[[38,39,40,41]]
                acc=cps[[42,43,44,45]]
                jerk=cps[[46,47,48,49]]
                strn = AndSal(100,h,*AndSal(100,k,*vel),*AndSal(100,k,*acc),*AndSal(100,k,*jerk),UnionCP(cps,1,9))

                vel=cps[[50,51,52,53]]
                acc=cps[[54,55,56,57]]
                jerk=cps[[58,59,60,61]]
                rhel=AndSal(100,h,*AndSal(100,k,*vel),*AndSal(100,k,*acc),*AndSal(100,k,*jerk),UnionCP(cps,7,11,18,21))

                vel=cps[[62,63,64,65]]
                acc=cps[[66,67,68,69]]
                jerk=cps[[70,71,72,73]]
                lhel=AndSal(100,h,*AndSal(100,k,*vel),*AndSal(100,k,*acc),*AndSal(100,k,*jerk),UnionCP(cps,6,10,19,20))

                vel=cps[[74,75,76,77]]
                acc=cps[[78,79,80,81]]
                jerk=cps[[82,83,84,85]]
                rplm=AndSal(100,h,*AndSal(100,k,*vel),*AndSal(100,k,*acc),*AndSal(100,k,*jerk),UnionCP(cps,5,13,23,25))

                vel=cps[[86,87,88,89]]
                acc=cps[[90,91,92,93]]
                jerk=cps[[94,95,96,97]]
                lplm=AndSal(100,h,*AndSal(100,2,*vel),*AndSal(100,2,*acc),*AndSal(100,2,*jerk),UnionCP(cps,4,12,22,24))

                margin_score = len(df.iloc[:,3].values)/100
                print(f'margin:{margin_score}')
                # unisco il tutto in un unico array
                final = UnionCPS(ariel,strn,rhel,lhel,rplm,lplm)
                #final = delnear(final,100)
                final = IgnoreZone(i,final,gt,margin_score)

                if dtwfilter:
                    final = FPremoverDTW(df.iloc[:,3],final,100)

                _,_,f1,_,_,_=f1scoremargin(gt,final,margin_score)
                if f1 < 0.5:
                    neg+=1
                f1_list.append(f1)
                PlotResult(df,np.array([]),final,"asd",margin_score,"",3)
                final = delnear(final,100)
                PlotResult(df,gt,final,timeseries[i],margin_score,"",3)
                    #print(f'f1:{round(f1,2)}->{round(f1_dtw,2)}  ---   {timeseries[i]} ')
                    #print(f'f1:{round(f1_no_dtw,2)}->{round(f1_no_dtw_dtw,2)}  ---   {timeseries[i]} ')
                    #print(f'f1:{round(f1_st_dtw,2)}->{round(f1_st_dtw_dtw,2)}  ---   {timeseries[i]} ')
            print(f'h:{h} k:{k} neg:{neg} mean:{sum(f1_list)/len(f1_list)} max:{max(f1_list)} min:{min(f1_list)} def')


            resexcel["k:"+str(k)+"_h:"+str(h)]= f1_list
    #resexcel.to_excel("outputFile/4compmodule_FPremoverDTW_before_normalized_standardized.xlsx")



# esperimento dove faccio un majority voting tra le feature e poi un delnear 
# da questo esperimento si evince che basta solo raggruppare nelle zone di densità maggiore le prediction
# andando a considerare tutti
def Together(cpsl,dtwfilter):
    cleaned_array = [s[3:-7] for s in timeseries]
    resexcel=pd.DataFrame()
    resexcel["name"] = cleaned_array
    for div in range(0,5):
        neg=0
        f1_list=[]
        for i in range(0,len(timeseries)):
            
            df=dfl[i]
            gt=gtl[i]
            cps=np.array(cpsl[i],dtype="object")
            #speed,acc,jerk,directnes,ke,density
            ariel_el=[26,27,28,29,30,31,32,33,34,35,36,37,2,8,14]
            ariel = UnionCPS(*cps[ariel_el])
            #speed,acc,jerk,ke,density
            strn_el=[38,39,40,41,42,43,44,45,46,47,48,49,1,9]
            strn = UnionCPS(*cps[strn_el])
            #speed,acc,jerk,ke,density,angle,angle
            rhel_el=[50,51,52,53,54,55,56,57,58,59,60,61,7,11,18,21]
            rhel = UnionCPS(*cps[rhel_el])

            lhel_el=[6,10,19,20,62,63,64,65,66,67,68,69,70,71,72,73]
            lhel = UnionCPS(*cps[lhel_el])
            
            rplm_el=[74,75,76,77,78,79,80,81,82,83,84,85,5,13,23,25]
            rplm = UnionCPS(*cps[rplm_el])

            lplm_el=[4,12,22,24,86,87,88,89,90,91,92,93,94,95,96,97]
            lplm = UnionCPS(*cps[lplm_el])
        
        
            ariel = MajorityVoteCP(ariel,100,div)#math.ceil(len(ariel_el)/div))
            strn = MajorityVoteCP(strn,100,div)#math.ceil(len(strn_el)/div))
            rhel = MajorityVoteCP(rhel,100,div)#math.ceil(len(rhel_el)/div))
            lhel = MajorityVoteCP(lhel,100,div)#math.ceil(len(lhel_el)/div))
            rplm = MajorityVoteCP(rplm,100,div)#math.ceil(len(rplm_el)/div))
            lplm = MajorityVoteCP(lplm,100,div)#math.ceil(len(lplm_el)/div))

            res = UnionCPS(ariel,strn,rhel,lhel,rplm,lplm)
            res = IgnoreZone(i,res,gt,100)
            res = delnear(res,100)
            if dtwfilter:
                res = FPremoverDTW(df.iloc[:,3],res,100)
                
            _,_,f1,_,_,_=f1scoremargin(gt,res,100)

            if f1 < 0.5:
                neg+=1
            f1_list.append(f1)


                #print(f'f1:{round(f1,2)}->{round(f1_dtw,2)}  ---   {timeseries[i]} ')
                #print(f'f1:{round(f1_no_dtw,2)}->{round(f1_no_dtw_dtw,2)}  ---   {timeseries[i]} ')
                #print(f'f1:{round(f1_st_dtw,2)}->{round(f1_st_dtw_dtw,2)}  ---   {timeseries[i]} ')

            #print(f'{timeseries[i]} f1:{f1}')
        print(f'neg:{neg} div:{div} mean:{sum(f1_list)/len(f1_list)} def')
        resexcel["div:"+str(div)]= f1_list


#resexcel.to_excel("outputFile/together_FPremoverDTW_before_normalized_standardized.xlsx")

"""
