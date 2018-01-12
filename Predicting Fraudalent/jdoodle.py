import pandas as pd
#import numpy as np
        
#Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Data exploration
train.describe()
train.shape
train.dtypes
train.isnull().sum()
len(train['transaction_id'].unique())

test.describe()
test.shape
test.dtypes
test.isnull().sum()
len(test['transaction_id'].unique())

#Data preprocessing

#Replacing NaN values to 'NA'
train = train.apply(lambda x:x.fillna('NA'))
test = test.apply(lambda x:x.fillna('NA'))

#Removing categories having only one level (either in train or test)
cols_to_remove = []
for c in train.columns:
    if c.startswith('cat'):
        if len(train[c].unique()) == 1:
            cols_to_remove.append(c)
            
for c in test.columns:
    if c.startswith('cat'):
        if len(test[c].unique()) == 1 and c not in cols_to_remove:
            cols_to_remove.append(c)
            
cols_to_remove.sort()

#Removing categories having two levels and having less than a count of 100 for any class (either in train or test)
for c in train.columns:
    if c.startswith('cat') and c not in cols_to_remove:
        if len(train[c].unique()) == 2  and any([count <= 100 for count in train[c].value_counts().values]):
            cols_to_remove.append(c)

for c in test.columns:
    if c.startswith('cat') and c not in cols_to_remove:
        if len(test[c].unique()) == 2  and any([count <= 100 for count in test[c].value_counts().values]):
            cols_to_remove.append(c)
            
cols_to_remove.sort()

#Replacing levels having frequency less than 1% with 'NA'
for c in train.columns:
    if c.startswith('cat') and c not in cols_to_remove:
        perc = train[c].value_counts(normalize = True)
        repl = perc.index[perc < 0.01]
        train.loc[train[c].isin(repl), c] = 'NA'
        
for c in test.columns:
    if c.startswith('cat') and c not in cols_to_remove:
        perc = test[c].value_counts(normalize = True)
        repl = perc.index[perc < 0.01]
        test.loc[test[c].isin(repl), c] = 'NA'
        
#Dropping varables having count less than 1% after replacing with 'NA'
for c in train.columns:
    if c.startswith('cat')and c not in cols_to_remove:
        if len(train[c].unique()) == 2 and any([count <= 0.01 for count in train[c].value_counts(normalize = True).values]):
            cols_to_remove.append(c)
            
for c in test.columns:
    if c.startswith('cat') and c not in cols_to_remove:
        if len(test[c].unique()) == 2 and any([count <= 0.01 for count in test[c].value_counts(normalize = True).values]):
            cols_to_remove.append(c)
cols_to_remove.sort()

#Creating X_train and y_train
X_train = train.drop(['transaction_id', 'target'] + cols_to_remove, axis=1)
y_train = train[['target']]
X_test = test.drop(['transaction_id'] + cols_to_remove, axis=1)
######################################################
#Converting to category
for c in X_train.columns:
   if c.startswith('cat'):
        X_train[c] = X_train[c].astype('category')
        cat_columns = X_train.select_dtypes(['category']).columns
        X_train[cat_columns] = X_train[cat_columns].apply(lambda x: x.cat.codes)

for c in X_test.columns:      
    if c.startswith('cat'):
        X_test[c] = X_test[c].astype('category')
        cat_columns1 = X_test.select_dtypes(['category']).columns
        X_test[cat_columns1] = X_test[cat_columns1].apply(lambda x: x.cat.codes)
        



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=9000, stratify = y_train)
    
#Scaling
from sklearn import preprocessing
for c in X_train.columns:
    scaler = preprocessing.StandardScaler()
    if X_train[c].dtype == 'float64':
        X_train[c] = scaler.fit(X_train[[c]]).transform(X_train[[c]])
#Apply train scaler on val and test
        X_val[c] = scaler.fit(X_train[[c]]).transform(X_val[[c]])
        X_test[c] = scaler.fit(X_train[[c]]).transform(X_test[[c]])
######################################################        
for c in X_val.columns:
    scaler = preprocessing.StandardScaler()
    if X_val[c].dtype == 'float64':
        X_val[c] = scaler.fit(X_val[[c]]).transform(X_val[[c]])
        
for c in X_test.columns:
    scaler = preprocessing.StandardScaler()
    if X_test[c].dtype == 'float64':
        X_test[c] = scaler.fit(X_test[[c]]).transform(X_test[[c]])

#Model development
cat_ind = [i for i in range(len(X_train.columns)) if X_train.iloc[:,i].dtype.name == 'category']

from catboost import CatBoostClassifier
clf = CatBoostClassifier(loss_function = 'Logloss', eval_metric ='AUC', use_best_model = True, nan_mode = 'Max').fit(X_train, y_train, cat_features =cat_ind, eval_set=(X_val, y_val))

print('Accuracy of GBM classifier on training set: {:.5f}'
     .format(clf.score(X_train, y_train)))
from sklearn.metrics import roc_auc_score
print('AUC of GBM classifier on training set: {:.5f}'
     .format(roc_auc_score(clf.predict(X_train), y_train)))
#Accuracy of GBM classifier on training set: 0.92894
#AUC of GBM classifier on training set: 0.94635
#Feature importance
print(clf.feature_importances_)
###################################################### 
#Predict on the test set
y_test = clf.predict_proba(X_test) 
#Submission
sub = pd.DataFrame({'transaction_id':test['transaction_id'],'target':y_test[:,1]})
sub = sub[['transaction_id','target']]
sub.to_csv('submissiongbm10.csv',index = False)
#AUC of GBM classifier on training set: 0.66868
