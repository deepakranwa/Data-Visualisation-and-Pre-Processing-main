# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:03:29 2020

@author: 
Ankit
B19236
9053437219
"""


#Q1=============================================================

print('\nQ1\n')

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('pima_indians_diabetes_miss.csv')
a=df.isnull().sum()         #returns sum of NaN values wrt each attribute
plt.bar(df.columns,a)
plt.xlabel('Attribute')
plt.ylabel('number of missing values')
plt.show()

#Q2================================================================
print('\n Q2(a)========================================= \n')

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('pima_indians_diabetes_miss.csv')   #row/index no. starts from 0
df['class']
a=len(df.columns)
b=0                                              #total tuple deleted
c=[]                                             #row numbers of the deleted tuple
for i in range(len(df.index)): 
    if (df.iloc[i].isnull().sum()) >= a/3:       #iloc gets rows at particulaer position in the index
        b=b+1
        c.append(i)
#df.drop(c)         
print('total tuple deleted:',b)
print('Row no. of deleted Tuple:','\n',*c )        


#Q2(b)=========Print In continuation ===========================
print('\n Q2(b)========================================== \n') 

df1 = df.drop(c)
d=len(df1.columns)
e=0                                              #total tuple deleted
f=[]                                             #row numbers of the deleted tuple
for i in df1.index : 
    if (df1['class'].loc[[i]].isnull().sum())>0:    #loc gets rows with particular labels from the index
        e=e+1
        f.append(i)
#df1.drop(f,inplace=True)                             #dropping null tuple('class')
print('total tuple deleted: ',e)
print('row no. of deleted Tuple:\n',*f ) 



#Q3================Print In continuation of Q2=======================================================
print('\nQ3=========================================== \n')

df2=df1.drop(f)
b=0
for i in df2:
    a=df2[i].isnull().sum()
    b=b+a
    print("number of missing values in",i,"is",a)
print("\nTotal no. of missing values",b )       
#Q4=====================================================
print('\nQ4========================================== \n')


df2=df1.drop(f)
import pandas as pd
import matplotlib.pyplot as plt
df_org=pd.read_csv('pima_indians_diabetes_original.csv')

def fill_missing_value(df2_):
    data=pd.concat((df2_.mean(),df2_.median(),df2_.mode().loc[0],
            df2_.mode().loc[1],df2_.std()),axis=1)         #axis=1 concatenate along row 
    data=data.T                                         #transposing data
    data.index=['Mean','Median','Mode1','Mode2','Statndard Dev']        #changing indexes
    data=data.T                                                         #again trans. data
    print('cleaned Data:\n')
    print(data)
    
    data_org=pd.concat((df_org.mean(),df_org.median(),df_org.mode().loc[0],
            df_org.mode().loc[1],df_org.std()),axis=1,)
    data_org=data_org.T
    data_org.index = ['Mean','Median','Mode1','Mode2','Statndard Dev']
    data_org=data_org.T
    print('\nOriginal Data:\n')
    print(data_org)
    
def RMSE(df2_):                           #root mean square error
    print('RMSE:\n')
    a=[]
    b=[]
    for i in df2.columns:
        RMSE=0
        a.append(i)
        null_index=df2[i][df2[i].isna()].index
        for j in null_index:
            RMSE+=(df2_[i][j]-df_org[i][j])**2      #squaring diff of replaced and orig value
        if len(null_index)>0:
            RMSE/=len(null_index)                                               
            RMSE**=0.5                              #finding RMSE 
            b.append(RMSE)
            print(i,':',RMSE)
        else:
            RMSE**=0.5
            b.append(RMSE)
            print(i,':',RMSE)
    plt.bar(a,b)
    plt.xlabel('Attribute')
    plt.ylabel('RMSE')
    plt.show()  
        
        
#a
print('4(a)=================================================')
df2_a=df2.fillna(df2.mean())
#i
print('a(i)\n')
fill_missing_value(df2_a)
#ii
print('\na(ii)\n')
RMSE(df2_a)
#b
print('\n4(b)===================================================')
df2_b=df2.interpolate()
#i
print('b(i)\n')
fill_missing_value(df2_b)
#ii
print('\nb(ii)\n')
RMSE(df2_b)




#Q5=====================================================
print('\nQ5========================================== \n')


def outlier(x):
    Q1,Q3=df2_b[x].quantile([0.25,0.75])
    top_whisker = Q3 + 1.5*(Q3-Q1)
    bottom_whisker = Q1-1.5*(Q3-Q1)
    outliers=pd.concat((df2_b[x][df2_b[x]> top_whisker],df2_b[x][df2_b[x]< bottom_whisker]))
    return outliers

def boxplot():
    fig,axs=plt.subplots(1,2,figsize=(9,6))
    axs[0].boxplot(df2_b['Age'],vert=True,patch_artist=True)
    axs[0].set_title('Age')
    axs[0].set_ylabel('Age')
    axs[1].boxplot(df2_b['BMI'],vert=True,patch_artist=True)
    axs[1].set_title('BMI')
    axs[1].set_ylabel('BMI')
    plt.show()
    
#i===================================================
print('i)\n')
print('Outliers in Age:')
print(*outlier('Age'))
print('\nOutliers in BMI:')
print(*outlier('BMI'))
boxplot()


#ii======================================================
print('ii)')
outliers_age=outlier('Age')
df2_b['Age'][outliers_age.index]=df2_b['Age'].median() #replacing outliers in 'Age'

outliers_bmi=outlier('BMI')
df2_b['BMI'][outliers_bmi.index]=df2_b['BMI'].median() #replacing outlier in 'BMI'
boxplot()

