#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


np.random.seed(0)


# In[5]:


X, y = make_blobs(n_samples=5000, centers=[[2,4], [-2,-1], [2,-3], [1,1]],  cluster_std=0.9)


# In[10]:


plt.scatter(X[:,0],X[:,1], marker='.')


# In[14]:


k_means=KMeans(init="k-means++", n_clusters=4, n_init=12)

k_means.fit(X)


# In[21]:


k_means_labels= k_means.labels_
k_means_labels

k_means_centers=k_means.cluster_centers_
k_means_centers


# In[24]:


fig = plt.figure(figsize=(6, 4))

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    my_members = (k_means_labels == k)

    cluster_center = k_means_centers[k]

    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)


ax.set_title('KMeans')

ax.set_xticks(())

ax.set_yticks(())


plt.show()


# In[28]:


import pandas as pd
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'
df=pd.read_csv(path)
df.head()


# In[37]:


df=df.drop('Address',axis=1)
df.head()


# In[36]:


from sklearn.preprocessing import StandardScaler
X=df.values[:, 1:]
X=np.nan_to_num(X)
cluster_df=StandardScaler().fit_transform(X)

cdf=cluster_df
cluster_df


# In[41]:


k_means=KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means.fit(cdf)
labels=k_means.labels_


# In[43]:


df['Cluster Kmean']=labels
df.head()


# In[46]:


df.groupby('Cluster Kmean').mean()


# In[48]:


area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


# In[51]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))

