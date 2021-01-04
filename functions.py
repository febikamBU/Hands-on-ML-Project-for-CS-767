#!/usr/bin/env python
# coding: utf-8

# In[4]:
# Importing libraries 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix,mean_absolute_error, classification_report,roc_auc_score, mean_squared_error
from time import time
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Evaluating & printing the model
def model_evaluation(y_test, y_pred):

    # Computing the following with confusion matrix
    cf_1 = confusion_matrix(y_test, y_pred)

    # Computing my simple model evaluation metrics - that is, TP, TN, FP etc.,
    TP = cf_1[1][1]  #TP - true positives
    FP = cf_1[0][1]  #FP - false positives
    TN = cf_1[0][0]  #TN - true negativess
    FN = cf_1[1][0]  #FN - false negatives
    TPR = round((TP/(TP + FN)) * 100, 2) #TPR = TP/(TP + FN)
    TNR = round((TN/(TN + FP)) * 100, 2) #TNR = TN/(TN + FP)
    ACC = round(((TP + TN)/(TP + TN + FP + FN)) * 100, 2)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = TP / (TP + FP)
    
    # Finding the RMSE value
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))    
    
    # Displaying the Performance metrics of the model
    print(f'TP - true positives: {TP}\nFP - false positives: {FP}\nTN - true negativess: '
          f'{TN}\nFN - false negatives: {FN}\nTPR - true positive rate: {TPR}%\n'
          f'TNR - true negative rate: {TNR}%')
    print("Mean Absolute Error (MAE) : {:.2}".format(mae))
    print("Root Mean Squared Error (RMSE) : {:.2}\n".format(rmse))

    print(classification_report(y_test, y_pred))
    print(end='\n')


# Method for DecisionTreeClassifier_model    
def decisionTreeClassifier_model(X_train, X_test, y_train, y_test):
    t0 = time()
    
    # Create classifier
    model = DecisionTreeClassifier(criterion='entropy')
    
    # Fit the classifier on the training features and labels.
    t0 = time()
    model.fit(X_train, y_train)
    print('\nPerfomance Report:\n')
    print("Training time:", round(time()-t0, 3), "s")

    # Predicting using X_test_norm
    t1 = time()
    y_pred = model.predict(X_test)
    print("Prediction time:", round(time()-t1, 3), "s\n")
 
    # Calling/displaying model evaluation function
    model_evaluation(y_test, y_pred)  
    

    return y_pred