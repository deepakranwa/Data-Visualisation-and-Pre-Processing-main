# -*- coding: utf-8 -*-
"""
@author: Harsh Arya
"""
#Contains different Data visualisation techniques 
#Q1=============================================================

print('\nQuestion 1:')
import pandas as pd   #importing pandas
df=pd.read_csv("landslide_data3.csv") 
dataframe=df.describe()
dataframe.drop(['count','25%','75%'],inplace=True)
frame=pd.concat((dataframe,df[df.columns[2:]].mode()))
frame.index=['Mean','Std Dev','Min','Median','Max','Mode']

print(frame[frame.columns[0:3]],'\n')
print(frame[frame.columns[3:6]],'\n')
print(frame[frame.columns[6:]])


   
#Q2(a)============================================================
print('\nQuestion 2:')
print('a)')
import matplotlib.pyplot as plt   #importing matplotlib
for i in df:
    if i != 'dates' and i!= 'stationid' and i!= 'rain':
        df.plot.scatter(x="rain",y=i)
plt.show()


print('b)')      #Scatter plot

import matplotlib.pyplot as plt 
for i in df:
    if i != 'dates' and i!= 'stationid' and i!= 'temperature':
        df.plot.scatter(x="temperature",y=i,color='orange')
plt.show()


#Q3==============================================================
print('\nQuestion 3:')          #correlation
print('a)')
print('Correlation of rain with:\n')    
r=df.corrwith(df['rain'])
print(r)

print('\nb)')
print('Correlation of temperature with:\n')
print(df.corrwith(df['temperature']))

#Q4=============================================================
print('\nQuestion 4:')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].hist(df['rain'])
axs[0].set_xlabel('Rainfall in ml')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Histogram of Rainfall')
axs[1].hist(df['moisture'])
axs[1].set_xlabel('Moisture in Percentage')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of Moisture')
plt.show()


#Q5==============================================================
print('\nQuestion 5:')
y=df.groupby('stationid')
fig,axs=plt.subplots(2,5,figsize=(22,10))
r=0;c=0
for i in y['rain']:
    axs[r][c].hist(i[1])
    axs[r][c].set_title('Sensor:'+i[0])
    axs[r][c].set_xlabel('Rain in ml')
    axs[r][c].set_ylabel('Frequency')
    c+=1
    if c==5:
        r=1;c=0
plt.show()

#Q6============================================================
print('\nQuestion 6:')
fig,axs=plt.subplots(2,1,figsize=(15,12))
axs[0].boxplot(df['rain'],vert=False,patch_artist=True)
axs[0].set_title('Rain')
axs[0].set_xlabel('Rain in ml')
axs[1].boxplot(df['moisture'],vert=False,patch_artist=True)
axs[1].set_title('Moisture')
axs[1].set_xlabel('Moisture in Percentage')
plt.show()

