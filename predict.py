import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils

DATA  = './data/validation.tsv'
MODEL = './model.h5'

def read_tsv_full(filename):
    df  = pd.read_csv(filename, header=None, delimiter='\t', names=['id','name','label','spectrum'])
    df = df.sample(frac=1,random_state=12345)
    X = []
    Y = []
    ids = []
    names = []
    for index, row in df.iterrows(): 
        Y.append( row['label'] == 'cancer' )
        X.append( np.array( row['spectrum'].split(',') ).astype(np.float) )
        ids.append( row['id'] )
        names.append( row['name'] )

    X = np.array(X)
    Y = np_utils.to_categorical(np.array(Y), 2).astype(np.int) ## BINARY! 
    return X,Y,ids,names

X,Y,ids,names = read_tsv_full( DATA )
model = load_model(MODEL)
#print(names)
#print(model.summary())
Y_pred = model.predict_proba(X, verbose=0)[:,0]
Y_real = Y[:,0]

stat = {}

for index in range( len(ids) ):
    Y_real_str = "skin"
    Y_pred_str = "skin"
    if ( Y_pred[index] >= 0.5 ): Y_pred_str = "cancer"
    if ( Y_real[index] == 1 ): Y_real_str = "cancer"
    Y_result = "correct"
    if ( Y_real_str == "skin" and Y_pred_str == "cancer" ):Y_result = "Type II: false alarm"
    if ( Y_real_str == "cancer" and Y_pred_str == "skin" ):Y_result = "Type I: missed target"

    strkey = names[index] + "_" + Y_real_str
    if strkey not in stat.keys(): stat[strkey] = { "total":0, "value":"", "correct":0}

    stat[ strkey ]["total"] += 1
    stat[ strkey ]["value"] = Y_real_str
    if Y_result == "correct": stat[ strkey ]["correct"] += 1

#    print("%s\t%s\t%.2f\t%s\t%s" % (
#        names[index],
#        Y_real_str,
#        Y_pred[index],
#        Y_pred_str,
#        Y_result
#    ))
#print("==================================\n")

tp = 0
tn = 0
fp = 0
fn = 0
for k in stat.keys():
    if stat[k]["value"] == "cancer" and stat[k]["correct"]*2 >= stat[k]["total"]:  tp += 1
    if stat[k]["value"] == "cancer" and stat[k]["correct"]*2 <  stat[k]["total"]:  fn += 1
    if stat[k]["value"] == "skin"   and stat[k]["correct"]*2 >= stat[k]["total"]:  tn += 1
    if stat[k]["value"] == "skin"   and stat[k]["correct"]*2 <  stat[k]["total"]:  fp += 1
# print("real = ", Y_real, " predicted:", Y_pred)

#print("tp:",tp,"tn:",tn,"fp:",fp,"fn:",fn)
print("total: %d, positives: %d, negatives: %d, tp: %d, fp: %d, tn: %d, fn: %d" % (tp+tn+fp+fn, tp+fn, tn+fp, tp, fp, tn, fn))
print("sensitivity: %.2f" % (tp/(tp+fn+0.0)))
print("specificity: %.2f" % (tn/(tn+fp+0.0)))
