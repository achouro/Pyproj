#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
df = pd.read_csv(path)
df.head()


# In[7]:


df['custcat'].value_counts()


# In[13]:


df.hist(column= 'income', bins=50)


# In[37]:


df.columns


# In[7]:


z = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values.astype(float)


# In[8]:


y=df[['custcat']]


# In[30]:


from sklearn.model_selection import train_test_split
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=4)
z_train.shape


# In[42]:


z_train_norm= preprocessing.StandardScaler().fit(z_train).transform(z_train.astype(float))
z_test_norm= preprocessing.StandardScaler().fit(z_test).transform(z_test.astype(float))


# In[33]:


from sklearn.neighbors import KNeighborsClassifier

k=4

neigh= KNeighborsClassifier(n_neighbors= k).fit(z_train_norm, y_train)
neigh


# In[40]:


yhat= KNeighborsClassifier(n_neighbors=k).fit(z_train_norm, y_train). predict(z_train_norm)

yhat


# In[49]:


from sklearn import metrics
print("Accuracy metrics for training set: \n",metrics.accuracy_score(y_train, neigh.predict(z_train_norm)))
print("Accuracy metrics for test set: \n",metrics.accuracy_score(y_test, neigh.predict(z_test_norm)))


# In[32]:


y=df[['custcat']]
z = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values

from sklearn.model_selection import train_test_split
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=4)

z_train_norm= preprocessing.StandardScaler().fit(z_train).transform(z_train.astype(float))
z_test_norm= preprocessing.StandardScaler().fit(z_test).transform(z_test.astype(float))


from sklearn.neighbors import KNeighborsClassifier

neig= KNeighborsClassifier(n_neighbors = 6).fit(z_train_norm,y_train)
yhat= neig.predict(z_train_norm)

yhat

import seaborn as sns
sns.scatterplot(x= 'income', y= 'custcat', color= 'b', data=df)


# In[74]:


import numpy as np
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

z_train_norm= preprocessing.StandardScaler().fit(z_train).transform(z_train.astype(float))
z_test_norm= preprocessing.StandardScaler().fit(z_test).transform(z_test.astype(float))



for i in range(1, Ks):
    
    neigh = KNeighborsClassifier(n_neighbors = i).fit(z_train_norm,y_train)
    yhat=neigh.predict(z_test_norm)

    mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)
    
  

mean_acc.argmax()
    

    
    


# In[2]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[11]:


plt.plot(y_hat, x)

