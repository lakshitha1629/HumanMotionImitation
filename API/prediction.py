import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
import warnings 
warnings.filterwarnings('ignore')

def fitModel(df,predictCol):
    df = df.dropna(how='any',axis=0) 
    # Get  columns whose data type is object i.e. string
    filteredColumns = df.dtypes[df.dtypes == np.object]
    # list of columns whose data type is object i.e. string
    listOfColumnNames = list(filteredColumns.index)
    cols = listOfColumnNames
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(columns=[predictCol], axis=1)
    y = df[predictCol]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('ET', ExtraTreesClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        cv_results = model_selection.cross_val_score(model, x_train, y_train,scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
#         print(msg)
    modelFit = model.fit(x_train, y_train)
    
    return modelFit, x_test, y_test

def createModel(model, x_test, y_test):
    # Save to file in the current working directory
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    # Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(x_test, y_test)
    # print("Test score: {0:.2f} %".format(100 * score))
    return pickle_model, score