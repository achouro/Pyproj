#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree


# In[4]:


path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
df=pd.read_csv(path)

df.head()


# In[7]:


df[["Drug"]].value_counts()


# In[9]:


df.shape


# In[39]:


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y= df[['Drug']]


# In[40]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


# In[41]:


le_BP= preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2]=le_BP.transform(X[:,2])


# In[42]:


le_Chol= preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3]=le_BP.transform(X[:,3])


# In[44]:


X[0:5]


# In[50]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=3)

(x_train.shape)


# In[58]:


from sklearn.tree import DecisionTreeClassifier
dectree= DecisionTreeClassifier(criterion='entropy', max_depth=4)

dectree.fit(x_train, y_train)

yhat=dectree.predict(x_train)


# In[125]:


le_drug= preprocessing.LabelEncoder()
le_drug.fit(['drugA','drugB','drugC','drugX','drugY'])
YY=le_drug.transform(Y)

x_train, x_test, yy_train, yy_test = train_test_split(X, YY, test_size= 0.3, random_state=3)


from sklearn.tree import DecisionTreeClassifier
dectree= DecisionTreeClassifier(criterion='entropy', max_depth=4)

dectree.fit(x_train, yy_train)


yyhat=dectree.predict(x_train)
yhat=dectree.predict(x_test)


yayo= [yyhat:yhat]


# In[126]:


YY.shape


# In[111]:


lyyhat=[]
for i in yyhat:
    lyyhat.append(i)
print(lyyhat)

lYY=[]
for i in YY:
    lYY.append(i)
print(lYY)




# In[118]:


def match(a,b):

    for i in :
        l=0
        if a[i]==b[i]:
            l= l+1
       
    print(l)
        
        
    
match(lyyhat, lYY)


# In[127]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[129]:


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


# In[131]:


tree.plot_tree(drugTree)
plt.show

