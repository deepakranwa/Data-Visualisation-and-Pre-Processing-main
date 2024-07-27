import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('mnist-tsne-train.csv')
test_data=pd.read_csv('mnist-tsne-test.csv')

from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

def Plot_PurityScore(prediction,data,x,N,cluster_centers=0):
    colour=['grey','tan','crimson','darkorange','gold','lawngreen','yellowgreen','cyan','steelblue','orchid',
            'lightgrey','wheat','olive','seagreen','teal','slategray','lavender','slateblue','fuchsia','pink','crimson']
    plt.figure(figsize=(10,10))
    
    for i in range(N):
        
        c=data[prediction==i]
        if len(c)==0:
            break
        plt.scatter(c[data.columns[0]],c[data.columns[1]],label='Cluster'+str(i),color=colour[i])
        if x!='DBSCAN':
            plt.scatter(cluster_centers[i][0],cluster_centers[i][1],marker='*',s=250,color='black')
    
    c=data[prediction==-1]
    if len(c!=0):
        plt.scatter(c[data.columns[0]],c[data.columns[1]],label='Noise',color=colour[i+1])
    
    plt.xlabel('Dimenson 1');plt.ylabel('Dimenson 2')
    plt.legend();plt.title(x);plt.show()
    
    print('Purity Score = ',end='')
    print(purity_score(data[data.columns[-1]],prediction))
    
#QUESTION 1
#K-Means
print('==============----Question1----=============','\n')

from sklearn.cluster import KMeans
K = 10
kmeans = KMeans(n_clusters=K,random_state=42)
kmeans.fit(train_data[train_data.columns[:-1]])

print('\nTraining Data:')
kmeans_prediction = kmeans.predict(train_data[train_data.columns[:-1]])
Plot_PurityScore(kmeans_prediction,train_data,'K-Means',K,kmeans.cluster_centers_)

print('\nTest Data:')
kmeans_prediction = kmeans.predict(test_data[test_data.columns[:-1]])
Plot_PurityScore(kmeans_prediction,test_data,'K-Means',K,kmeans.cluster_centers_)

#QUESTION 2
#GMM
print('==============----Question2----=============','\n')


from sklearn.mixture import GaussianMixture
K = 10
gmm = GaussianMixture(n_components = K,random_state=42)
gmm.fit(train_data[train_data.columns[:-1]])

print('\nTraining Data:')
GMM_prediction = gmm.predict(train_data[train_data.columns[:-1]])
Plot_PurityScore(GMM_prediction,train_data,'GMM',K,gmm.means_)

print('\nTest Data:')
GMM_prediction = gmm.predict(test_data[test_data.columns[:-1]])
Plot_PurityScore(GMM_prediction,test_data,'GMM',K,gmm.means_)

#QUESTION 3
#DBSCAN
print('==============----Question3----=============','\n')

from sklearn.cluster import DBSCAN
dbscan_model=DBSCAN(eps=5, min_samples=10).fit(train_data[train_data.columns[:-1]])

print('\nTraining Data:')
DBSCAN_predictions = dbscan_model.labels_
Plot_PurityScore(DBSCAN_predictions,train_data,'DBSCAN',len(set(DBSCAN_predictions)))


from scipy import spatial as spatial
def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
# Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
# Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
# Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
# Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break                
    return y_new
DBSCAN_predictions = dbscan_predict(dbscan_model, test_data[test_data.columns[:-1]].values, metric =
 spatial.distance.euclidean)

print('\nTest Data:')
Plot_PurityScore(DBSCAN_predictions,test_data,'DBSCAN',len(set(DBSCAN_predictions)))

x=input('\nPRESS THE ENTER KEY FOR BONUS QUESTIONS')


#BONUS QUESTIONS
print('=========----BONUS QUESTIONS: ----=========','\n')

#A
K=[2, 5, 8, 12, 18, 20]
Distortion=[]
from scipy.spatial.distance import cdist
for k in K:
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(train_data[train_data.columns[:-1]])
    print('\n\t\t\tK =',k)
    print('\nTraining Data:')
    kmeans_prediction = kmeans.predict(train_data[train_data.columns[:-1]])
    Plot_PurityScore(kmeans_prediction,train_data,'K-Means',k,kmeans.cluster_centers_)

    print('\nTest Data:')
    kmeans_prediction = kmeans.predict(test_data[test_data.columns[:-1]])
    Plot_PurityScore(kmeans_prediction,test_data,'K-Means',k,kmeans.cluster_centers_)
    
    Distortion.append(sum(np.min(cdist(train_data[train_data.columns[:-1]].values, kmeans.cluster_centers_, 'euclidean'), axis=1)) / train_data.shape[0])
plt.figure(figsize=(10,10))
plt.plot(K,Distortion)
plt.scatter(K,Distortion,marker='*',color='r')
plt.xlabel('K');plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()

K=[2, 5, 8, 12, 18, 20]
Loglikelihood=[]
for k in K:
    gmm = GaussianMixture(n_components = k,random_state=42)
    gmm.fit(train_data[train_data.columns[:-1]])
    print('\n\t\t\tK =',k)
    print('\nTraining Data:')
    GMM_prediction = gmm.predict(train_data[train_data.columns[:-1]])
    Plot_PurityScore(GMM_prediction,train_data,'GMM',k,gmm.means_)
    
    print('\nTest Data:')
    GMM_prediction = gmm.predict(test_data[test_data.columns[:-1]])
    Plot_PurityScore(GMM_prediction,test_data,'GMM',k,gmm.means_)
    
    Loglikelihood.append(gmm.lower_bound_)
plt.figure(figsize=(10,10))
plt.plot(K,Loglikelihood)
plt.scatter(K,Loglikelihood,marker='*',color='r')
plt.xlabel('K');plt.ylabel('Loglikelihood')
plt.title('Elbow Method')
plt.show()  


