#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
df=pd.read_csv(path)

df[["Drug"]].value_counts()
df.head()

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y= df[['Drug']]

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP= preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2]=le_BP.transform(X[:,2])

le_Chol= preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3]=le_BP.transform(X[:,3])


from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

tree.plot_tree(drugTree)
plt.show


# In[8]:


0.1*sum([79353.29, 41461.30, 79595.95, 45390.47, 86609.11, 87360.22, 62919.22, 58086.61, 78350.61, 44705.00])


# In[11]:


import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"

data = pd.read_csv(URL)
data.head()


# In[17]:


data.dropna(inplace=True)
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)


# In[27]:


regtree= DecisionTreeRegressor(criterion='squared_error')
regtree.fit(X_train, Y_train)
pred= regtree.predict(X_test)
regtree.score(X_test, Y_test)

print("Mean erra iz", (pred - Y_test).abs().mean()*1000, "dolla")

