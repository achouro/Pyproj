#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[5]:


path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

df=pd.read_csv(path)

df.head()


# In[8]:


df =df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
df['churn'] = df['churn'].astype('int')
df.head()


# In[23]:


#define
X= np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
Y= np.asarray(df[['churn']])

#normalise
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)


#split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state=4)

#model LR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR= LogisticRegression(C=0.01, solver='liblinear').fit(X_train, Y_train)

yhat=LR.predict(X_test)

#probality
prob_yhat= LR.predict_proba(X_test)
prob_yhat[:10]


# In[38]:


from sklearn.metrics import jaccard_score

jaccard_score(Y_test, yhat, pos_label=0)



# In[46]:


#diagnostics
from sklearn.metrics import classification_report, confusion_matrix, log_loss

cr=classification_report(Y_test, yhat)

cm= confusion_matrix(Y_test, yhat, labels=[1,0])

ll= log_loss(Y_test, prob_yhat)

print("Confusion matrix: \n",cm,"\n","\n", "Classification report: \n", cr ,"\n","\n","Log loss: \n", ll)


# In[47]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[50]:


path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"
cdf= pd.read_csv(path)
cdf.head()


# In[53]:


#cleaning
cdf = cdf[pd.to_numeric(cdf['BareNuc'], errors='coerce').notnull()]
cdf['BareNuc'] = cdf['BareNuc'].astype('int')
cdf['Class'] = cdf['Class'].astype('int')


# In[57]:


#preprocessing
X= np.asarray(cdf[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
Y= np.asarray(cdf[['Class']])

#split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y,test_size=0.2 ,random_state=4)

#model SVM

from sklearn import svm

SVM= svm.SVC(kernel='rbf')
SVM.fit(X_train, Y_train)

yhat=SVM.predict(X_test)

yhat[:5]


# In[60]:


#diagnostics

from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score

cr=classification_report(Y_test, yhat)

cm= confusion_matrix(Y_test, yhat, labels=[2,4])

f1= f1_score(Y_test, yhat, average='weighted') 

jc= jaccard_score(Y_test, yhat,pos_label=2)

print("Confusion matrix: \n",cm,"\n","\n", "Classification report: \n", cr ,
      "\n","\n","F1 Score: \n", f1,"\n","\n","Jaccard Score: \n", jc)

