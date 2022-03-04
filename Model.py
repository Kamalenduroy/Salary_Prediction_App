# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 04:11:25 2022

@author: rkama
"""
#Importing Libraries
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pickle

#Reading dataset
dataset = pd.read_csv('Salary.csv')

#Cleaning the data
dataset['Experience'].fillna(0,inplace=True)

dataset['Test_Score'].fillna(dataset['Test_Score'].mean(),inplace=True)

word_dict = {'one':1, 'two ':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10, 'eleven':11,0:0}

dataset['Experience'] = dataset['Experience'].apply(lambda x: word_dict[x])

dataset.drop(columns='Index',axis=1,inplace=True)

#Creating independent and dependent variables
y = dataset.pop('Salary')
X = dataset

#Training Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

#Dumping the model into a file using pickle
pickle.dump(lr,open('model.pkl','wb'))

#Loading the file into model again for verification
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5,6,7]]))

print(X)