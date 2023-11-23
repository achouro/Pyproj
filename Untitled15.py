#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[106]:


filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

df=pd.read_csv(filepath, header=None)


# In[107]:


head_name=['age', 'gender', 'bmi', 'no_children', 'smoker', 'region', 'charges'] 

df.columns=head_name
df.replace( "?", np.NaN, inplace=True)
df.head(10)


# In[108]:


df.info()


# In[109]:


mean_age= df['age'].astype('float').mean(axis=0)
df['age'].replace( np.NaN, mean_age, inplace=True)

most_frequent_smok= df['smoker'].value_counts().idxmax()
df['smoker'].replace( np.NaN, most_frequent_smok, inplace=True)

df[['age', 'smoker']]= df[['age', 'smoker']].astype('int')

df['charges']=np.round(df['charges'], 2)
df.info()


# In[110]:


sns.regplot(x="bmi", y="charges", data=df, )


# In[111]:


sns.boxplot(x="smoker", y="charges", data=df, )


# In[47]:


df.corr()


# In[112]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(df[['smoker']], df[['charges']])
yhat_lr=lr.predict(df[['smoker']] )
yhat_lr


# In[113]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()



# In[117]:


Y=df[['charges']]

Z = df[["age", "gender", "bmi", "no_children", "smoker", "region"]]

lr.fit(Z,Y)

lr.score(Z, Y)


# In[125]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# In[128]:


Input= [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]

pipe=Pipeline(Input)
pipe.fit(Z,Y)

pipe.predict(Z)

pipe.score(Z,Y)


# In[131]:


from sklearn.model_selection import train_test_split
Z_train, Z_test, Y_train, Y_test = train_test_split(Z,Y, test_size=0.2, random_state=1)


# In[134]:


from sklearn.linear_model import Ridge
RR= Ridge(alpha= 0.1)

RR.fit(Z,Y)

RR.score(Z,Y)


# In[138]:


pr=PolynomialFeatures(degree=2)
Z_train_pr= pr.fit_transform(Z_train)
Z_test_pr= pr.fit_transform(Z_test)

RR.fit(Z_train_pr,Y_train)
RR.score(Z_train_pr,Y_train)

