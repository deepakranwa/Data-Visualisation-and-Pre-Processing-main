# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:54:59 2020

@author: ankit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


df=pd.read_csv('datasetA6_HP.csv')
x_t=df['HP']


#Question_1
print('==============----Question1----=============','\n')

#a)
print('\n----a)----','\n')
    
df['Date']=pd.to_datetime(df['Date'],dayfirst=True)
x=df['Date']
y=df['HP']
plt.plot(x,y)
plt.xticks(rotation='45')   #inclination of X ticks
plt.xlabel('Date')
plt.ylabel('Power in MU')
plt.show()


#b)
print('\n----b)----','\n')

x_t_1=df['HP'][:-1] #time sequence with 1 day lag
auto_corr = np.corrcoef(x_t[1:],x_t_1)[0][1]   #Pearson correlation with 1 day time lag
print('Correlation coefficient between x_t and x_t-1 : ',end='')
print(auto_corr)

#c)
print('\n----c)----','\n')
x_t_1=df['HP'][:-1] #time sequence with 1 day lag
plt.scatter(x_t[1:],x_t_1,marker='.')      #generating Scatter plot b/w given and 1 day time lag
plt.xlabel('Given Time Sequence')
plt.ylabel('One-Day Lagged Generated Sequence')
plt.show()


#d)
print('\n----d)----\n')
x_t=df['HP']
Correlation=[]
for p in range(1,8):
    x_t_p=df['HP'][:-p]  #sequence with p day time lag
    corr_coef=np.corrcoef(x_t[p:],x_t_p)[0][1]   #finding correlation coef from corr matrix
    Correlation.append(corr_coef) 
    print('Correlation Coefficient for lag={} is'.format(p),corr_coef)
plt.plot(range(1,8),Correlation)
plt.scatter(range(1,8),Correlation,marker='o')    #scatter plot b/w p lag and corresponding correlation 
plt.ylabel('Correlation Coefficient')
plt.xlabel('Lagged Value')
plt.title('Autocorrelation')
plt.show()

#e)
print('\n----e)----','\n')
sm.graphics.tsa.plot_acf(df['HP'],lags=range(1,8))  #using plot_acf fn for same as d
plt.ylabel('Correlation Coefficient')
plt.xlabel('Lagged Value')
plt.show()


#Question_2
print('\n==============----Question2----=============')

#persistence model
train, test= x_t[1:len(x_t)-250], x_t[len(x_t)-250:]
test_t=test.values[:-1]
test_t_1=test.values[1:]
print('\nRMSE of Persistance Model : ',end='')
print(((test_t-test_t_1)**2).mean()**0.5)       #RMSE for this model



#Question_3
print('\n==============----Question3----=============')

#a)
print('\n----a)----','\n')
    
train, test= x_t[1:len(x_t)-250], x_t[len(x_t)-250:]
model=AR(train)     #Using AutoRegression Model
model_fit = model.fit(maxlag=5)   #fitting the time lag as 5
#Make prediction using AR model
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)   
rmse=((prediction-test)**2).mean()**0.5   #RMSE for AR model
print('\nRMSE of AR(5) Model : ',end='')
print(rmse)

plt.scatter(test,prediction)   #Scatter plot b/w original and predicted 
plt.xlabel('Original Test Data')
plt.ylabel('Predicted Test Data')
plt.title('AR(5)')
plt.show()


#b)
print('\n----b)----','\n')
    
print('p:\t RMSE:\n')
Lag=[1,5,10,15,25]
RMSE=[]
for p in Lag:
    model=AR(train)         #AR model for different Time Lag
    model_fit = model.fit(maxlag=p)
    prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    rmse=((prediction-test)**2).mean()**0.5    #RMSE for different time Lag in AR model
    RMSE.append(rmse)
    print(p,'\t',rmse)


#c)
print('\n----c)----','\n')

x_t=train
for h in range(1,len(train)):
    x_t_h=train[h:]   #Computeing the heuristic value for optimal number using given codition 
    if abs(np.corrcoef(x_t_h,x_t[:-h])[0][1]) < 2/len(train)**0.5:   #condition
        h=h-1           #optimal number
        print('Heuristic Value for Optimal Number of Lags :',h)
        break

model=AR(train.values)      #Using AR model for Heurestic value 
model_fit = model.fit(h)
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse=((prediction-test)**2).mean()**0.5
print('\nRMSE for Optimal Lags : ',end='')
print(rmse)



#d)
print('\n----d)----','\n')

print('\nUsing Heuristics for Calculating Optimal Lag:')
print('\np:\t RMSE:')
print(h,'\t',rmse)

print('\nWithout using Heuristics for Calculating Optimal Lag:')
print('\np:\t RMSE:')
optimal_index=RMSE.index(min(RMSE))
print(Lag[optimal_index],'\t',RMSE[optimal_index])





