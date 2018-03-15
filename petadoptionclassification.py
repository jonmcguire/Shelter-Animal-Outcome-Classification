# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:35:39 2018

@author: Jon
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#load data
train= pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


#ageuponoutcome to numerical in weeks
def agetodays(data):
        try:
            y = data.split()
        except:
            return None 
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365/12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])
        
train['AgeInDays'] = train['AgeuponOutcome'].apply(agetodays)
test['AgeInDays'] = test['AgeuponOutcome'].apply(agetodays)


#datetime into hour and month

train['hours'] = train.DateTime.str[11:13].astype('int')
train['month'] = train.DateTime.str[5:7].astype('int')
test['hours'] = test.DateTime.str[11:13].astype('int')
test['month'] = test.DateTime.str[5:7].astype('int')

#visualization
sns.countplot(x='OutcomeType',data=train, hue='AnimalType')

sns.countplot(x='AnimalType',data=train)

sns.countplot(x='hours', hue='AnimalType',data=train)
sns.countplot(x='month', hue='AnimalType',data=train)


#check for missing values
train.isnull().sum()

#replace nan in hours
train['AgeInDays'].fillna(0, inplace=True)
test['AgeInDays'].fillna(0, inplace=True)

#map to cut down on variables in columns
neuter = {'Neutered Male':1, 'Spayed Female':1, 'Intact Male':0, 'Intact Female':0, 'Unknown':0, np.nan:0} 
train['Neutered'] = train['SexuponOutcome'].map(neuter)

sex = {'Neutered Male':1, 'Spayed Female':0, 'Intact Male':1, 'Intact Female':0, 'Unknown':0, np.nan:0} 
train['Sex'] = train['SexuponOutcome'].map(sex)

out = {'Return_to_owner':3, 'Euthanasia':2, 'Adoption':0, 'Transfer':4, 'Unknown':3,'Died':1, np.nan:3} 
train['OutcomeType'] = train['OutcomeType'].map(out)

#test
test['Neutered'] = test['SexuponOutcome'].map(neuter)
test['Sex'] = test['SexuponOutcome'].map(sex)



#set y
y_train=train['OutcomeType']


#drop columns from X
train.drop(train.columns[[0,1,2,3,4,6,7,8,9]], axis=1, inplace=True)
test.drop(test.columns[[0,1,2,4,5,6,7]], axis=1, inplace=True)

#onehotencode
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#X and y seperation
X_train=train.iloc[:,:].values
X_test=test.iloc[:,:].values

#training
mod=GaussianNB()
mod.fit(X_train,y_train)

#optimize random forest
#n_estimators = 100
#optgraph=[]
#forest = RandomForestClassifier(warm_start=True, oob_score=True)

#for num in range(1, n_estimators + 400):
#    forest.set_params(n_estimators=num)
#    forest.fit(X_train, y_train)
#    print(num)
#    optgraph.append(forest.oob_score_)

#sns.distplot(optgraph, hist=True, kde=False)
#max(optgraph)


#mod = RandomForestClassifier(n_estimators = 415, max_features='auto')
mod.fit(X_train,y_train)

#testing
#y_hat=mod.predict(X_test)

#output
#cm=confusion_matrix(y_test,y_hat)
#cm

prob=mod.predict_proba(X_test)
output=pd.DataFrame(prob)
output.columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer']
output.to_csv("ouput.csv")
