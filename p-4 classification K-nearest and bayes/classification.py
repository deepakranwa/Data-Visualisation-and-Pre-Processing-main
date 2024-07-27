# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:35:12 2020

@author: ankit
"""
# Data Classification using K-Nearest Neighbor Classifier and Bayes 
#Classifier with Unimodal Gaussian Density

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

columns=['seismic','seismoacoustic','shift','genergy','gpuls',
         'gdenergy','gdpuls','ghazard','energy','maxenergy','class']
df=pd.read_csv('seismic_bumps1.csv',usecols=columns)  #only selected columns

#Question_1
print('==============----Question1----=============','\n')

[X_train, X_test, X_label_train, X_label_test] =train_test_split(df[df.columns[:-1]],   
                            df['class'], test_size=0.3, random_state=42,shuffle=True) 
#X_train/test=  from df excluding last class column
#X_label_train/test from the class column
#test_size give the size of the test examples(X,X_label) from the data
#random_state controls the ordering/shuffling(same for both x and class)
#shuffle=True, shuffling before the split

pd.concat((X_train,X_label_train),axis=1).to_csv('seismic_bumps_train.csv',index=False)
pd.concat((X_test,X_label_test),axis=1).to_csv('seismic_bumps_test.csv',index=False)
#Saving the train and test data to a new files , axis=1 for column

Acc_rate=[]
def KNN(X_train,X_label_train,X_test):  #K_neighbor_classifier
    acc_score=0
    print('\n Confusion Matrix:  |\tAccuracy score:  |\tk:\n')
    for k in [1,3,5]:
        knn=KNeighborsClassifier(n_neighbors=k)     #Finds the K-neighbors.
        knn.fit(X_train,X_label_train)  #Fitting the model using training data and target(labelled) values    
        Predictor=knn.predict(X_test)   #Predict the class labels for the provided data.
        
        #compute a confusion matrix to evaluate the accuracy of a classification 
        print(metrics.confusion_matrix(X_label_test,Predictor),end='\t\t')
        #finding accuracy of the kNN((TP+TN)/total samples used for testing),rounding the value upto 5 decimal 
        print(round(metrics.accuracy_score(X_label_test,Predictor),5),end='\t\t\t')
        print(k,'\n')
        if metrics.accuracy_score(X_label_test,Predictor) > acc_score:
            acc_score=round(metrics.accuracy_score(X_label_test,Predictor),5)
            k_best=k   #K for which accuracy is highest
    Acc_rate.append(acc_score)
    print('At k =',k_best,'accuracy is high at:',acc_score,'\n')

KNN(X_train,X_label_train,X_test)  #calling KNN function

#Question_2

print('==============----Question2----=============','\n')

X_test=(X_test-X_train.min())/(X_train.max()-X_train.min())  #normalising test example
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())  #normalising train example

pd.concat((X_train,X_label_train),axis=1).to_csv('seismic_bumps_train_normalized.csv')
pd.concat((X_test,X_label_test),axis=1).to_csv('seismic_bumps_test_normalized.csqv')

KNN(X_train,X_label_train,X_test)   #calling KNN function after normalization


#question_3

print('==============----Question3----=============','\n')

X_train=pd.read_csv('seismic_bumps_train.csv')
X_test=pd.read_csv('seismic_bumps_test.csv')

#finding mean vector and covariance matrix for both classes
Class0=X_train[X_train['class']==0][X_train.columns[0:-1]]  #training ex.of class0 
Mean_Class0=Class0.mean().values    #mean vector class1
#print(Mean_Class0)
Cov_Class0=Class0.cov().values      #covariance matrix for class0
#print(Cov_Class0)


Class1=X_train[X_train['class']==1][X_train.columns[0:-1]]  #training ex.of class1
Mean_Class1=Class1.mean().values    #mean vector class0
#print(Mean_Class1)
Cov_Class1=Class1.cov().values      #covariance matrix for class1
#print(Cov_Class1)

#finding Prior(P(Ci)) INFO for both classes
P_Class0=len(Class0)/(len(Class0)+len(Class1))    #Prior class0
P_Class1=len(Class1)/(len(Class0)+len(Class1))    #Prior class1
d=len(X_test.columns)-1            #dimension of test example

Predicted_class=[]     #predicting class of a test eg.
for x in X_test[X_test.columns[0:-1]].values:
    
    #finding likelihood of both classes P(x|Ci)
    p_x_Class0=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_Class0)**0.5)*np.e**(-0.5*np.dot(np.dot((x-Mean_Class0).T,np.linalg.inv(Cov_Class0)),(x-Mean_Class0)))
    p_x_Class1=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_Class1)**0.5)*np.e**(-0.5*np.dot(np.dot((x-Mean_Class1).T,np.linalg.inv(Cov_Class1)),(x-Mean_Class1)))
    #Evidence that test sample exist(P(x))
    P_x=p_x_Class0*P_Class0+p_x_Class1*P_Class1
    
    #finding posterior probability((likelihood*prior)/evidence)
    P_Class0_x=p_x_Class0*P_Class0/P_x
    P_Class1_x=p_x_Class1*P_Class1/P_x
    
    #comparing posterior prob of both class and assigning max value
    if P_Class0_x>P_Class1_x:
        Predicted_class.append(0)
    else:
        Predicted_class.append(1)

print('Confusion Matrix :  |\tAccuracy score :')
 #compute a confusion matrix to evaluate the accuracy of a bayes classiffier
print(metrics.confusion_matrix(X_test[X_test.columns[-1]],Predicted_class),end='\t\t')
#finding accuracy of the Bayes classifier ((TP+TN)/total samples used for testing)
print(round(metrics.accuracy_score(X_test[X_test.columns[-1]],Predicted_class),5),'\n') 
#rounding the value upto 5 decimal

Acc_rate.append(round(metrics.accuracy_score(X_test[X_test.columns[-1]],Predicted_class),5))
#print(Acc_rate)    #Try printing Acc_score of all the method


#question_4

print('==============----Question4----=============','\n')
Best_result=pd.DataFrame(Acc_rate,index=['KNN','KNN on normalised data','Bayes'],columns=['Accuracy:'])
print(Best_result)





























