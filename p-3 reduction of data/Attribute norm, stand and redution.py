# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:05:16 2020

@author: ankit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df=pd.read_csv('landslide_data3.csv')
df.drop(['dates','stationid'],axis=1,inplace=True)

#1

print('==========----Question 1:------==========')

pd.options.mode.chained_assignment=None   #disabling warning of editing dataframe
n=0                 #Initializing attribute count as 0
for i in df.columns:
    Q1,Q3=df[i].quantile([0.25,0.75])
    iqr=Q3-Q1
    top_whisker = Q3 + 1.5*(iqr)
    bottom_whisker = Q1-1.5*(iqr)
    outlier=(df[i]<bottom_whisker) | (df[i]>top_whisker) 
    #print("\nNumber of outliers in",i,":", outlier.sum())          #try printing outliers
    outliers=df[i][outlier]      #Dataframe of outliers in df[i]
    df[i].iloc[outliers.index]=(df[i].drop(outliers.index)).median() #Replacing outliers with median of remaning values of df[i]
    n+=1                         #Incrementing attribute count


#a
print('a)')
df1=df.copy()
pd.options.mode.chained_assignment=None   #disabling warning of editing dataframe
def Min_Max():          #Function for computing Minimum and Maximum
    min_max=pd.concat((df1.min(),df1.max()),axis=1) #Dataframe with min and max of Df
    min_max=min_max.T           #Transpose
    index=pd.Series(['Min','Max'])
    min_max.set_index([index],inplace=True)     #Setting Index
    min_max=min_max.T
    print(min_max)
print('\nBefore Min_Max Normalization:')
Min_Max() 

df1=(df-df.min())/(df.max()-df.min())*(9-3)+3   #Min_Max Normalization
print('\nAfter Min_Max Normalization:')
Min_Max() 


#b
print('\nb)')
df2 = df.copy()
pd.options.mode.chained_assignment=None   #disabling warning of editing dataframe
def Standardization():  #Function for computing Mean and Standard Deviation 
    mean_std=pd.concat((round(df2.mean(),6),df2.std()),axis=1)#Dataframe with Mean and Std Dev of Df
    mean_std=mean_std.T
    index=pd.Series(['Mean','Standard Dev'])
    mean_std.set_index([index],inplace=True) #Setting Index
    mean_std=mean_std.T
    print(mean_std)
print('\nBefore Standardization:')
Standardization()

df2 = (df-df.mean())/(df.std())
print('\nAfter Standardization:')
Standardization() 


#=========================------Question2-------====================

print('\n======================------Question 2-------==================')
mean=np.array([0,0])
cov=np.array([[5,10],[10,13]])
D=np.random.multivariate_normal(mean,cov,1000,'ignore') #Generating Data
#print(np.cov(D.T)  #cov matrix from generated data

#a
print('a)')
x1=D.T[0]    
x2=D.T[1]       
plt.scatter(x1,x2,marker='x',color='blue')
plt.xlim(-15,15);plt.ylim(-15,15)
plt.xlabel('x1');plt.ylabel('x2')
plt.title('scatter Plot of 2D synthetic data')
plt.show()


#b
print('\n--------------b)-----------------')
eigval,eigvec=np.linalg.eig(np.cov(D.T))
print('Eigen values:',*eigval,'\nEigen vectors:',*eigvec.T) 

plt.figure(figsize=(6,6))
x1=D.T[0] ; x2=D.T[1]
plt.scatter(x1,x2,marker='x',color='blue')
plt.quiver([0],[0],eigvec[0][0],eigvec[1][0],angles="xy",color='red',scale=6)
plt.quiver([0],[0],eigvec[0][1],eigvec[1][1],angles="xy",color='red',scale=3)
plt.xlabel('x1');plt.ylabel('x2')
plt.xlim(-15,15);plt.ylim(-15,15)
plt.axis('equal')
plt.title('Plot of 2D synthetic data and Eigen vectors')
plt.show()


#c
print('\n----------------c)-----------\n')

prj=np.dot(D,eigvec)        #DOT product(projection) of 1000*2 with 2*2


plt.figure(figsize=(6,6))
x1=D.T[0] ; x2=D.T[1]
plt.scatter(x1,x2,marker='x',color='blue')
plt.quiver([0],[0],eigvec[0][0],eigvec[1][0],angles="xy",color='red',scale=6)
plt.quiver([0],[0],eigvec[0][1],eigvec[1][1],angles="xy",color='red',scale=3)
l1=prj[:,0]*eigvec[0][0]        #giving direction 
l2=prj[:,0]*eigvec[1][0]
plt.scatter(l1,l2 , color='magenta' , marker='x')
plt.xlabel('x1');plt.ylabel('x2')
plt.xlim(-15,15);plt.ylim(-15,15)
plt.title('Projected values on 1st eigen vector')
plt.axis('equal')
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(D.T[0],D.T[1],marker='x',color='blue')
plt.quiver([0],[0],eigvec[0][0],eigvec[1][0],angles="xy",color='red',scale=6)
plt.quiver([0],[0],eigvec[0][1],eigvec[1][1],angles="xy",color='red',scale=3)
plt.scatter(prj[:,1]*eigvec[0][1],prj[:,1]*eigvec[1][1],color='magenta',marker='x')
plt.xlabel('x1');plt.ylabel('x2')
plt.xlim(-15,15);plt.ylim(-15,15)
plt.title('Projected values on 2nd eigen vector')
plt.axis('equal')
plt.show()


#d
print('\n---------------------------d)-------------------------')
D_=np.dot(prj,eigvec.T)#Reconstructing Reduced Dimensional Data
print('Reconstructional Error:',(((D-D_)**2).sum()/len(D_)))




#3
print('\n===================-------Question 3:------------================')
df = df2 #Replacing df with standardized df

#a
print('a)')
eigval,eigvec=np.linalg.eig(np.cov(df.T))#Eigen Value and Eigen Vector
eigval.sort()        #Sorting Eigen Values in Ascending Order
eigval=eigval[::-1]  #Reversing (In Descending Order)


pca=PCA(n_components=2)         #PCA with l=2
Data=pca.fit_transform(df)


for i in range(2):
    print('Variance along Eigen Vector',i+1,':',np.var(Data.T[i]),
          '\nEigen Value corresponding to Eigen Vector',i+1,':',eigval[i],'\n')


plt.scatter(Data.T[0],Data.T[1])# Scatter plot of reduced dimensional data.
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter plot of reduced dimension data')
plt.show()


#b
print('\n---------b)--------')
plt.bar(range(1,8),eigval,color='orange')
plt.plot(range(1,8),eigval,color='black')
plt.scatter(range(1,8),eigval)
plt.xlabel('Index');plt.ylabel('Eigen Value')
plt.title('Eigen Values in Descending Order')
plt.show()


#c
print('c)')
RMSE=[] #Empty List
for i in range(1,8):
    pca=PCA(n_components=i)        #PCA with l=i
    Data=pca.fit_transform(df)     #Data with Reduced Dimension
    D_=pca.inverse_transform(Data) #Reconstructed Data
    RMSE.append((((df.values-D_)**2).mean())**.5) #Appending list with RMSE
    
plt.bar(range(1,8),RMSE,color='yellow') #Plot
plt.plot(range(1,8),RMSE,color='black')
plt.scatter(range(1,8),RMSE)
plt.ylabel('RMSE');plt.xlabel('l')
plt.title('Reconstruction Error')
plt.show()





