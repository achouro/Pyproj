#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
df.head()


# #Q1
# df.dtypes

# In[96]:


#Q1
df.dtypes


# In[26]:


df.describe()


# In[30]:


#Q2
df.drop('Unnamed: 0', axis=1, inplace= True)
df.drop('id', axis=1, inplace=True)


# In[32]:


df.describe()


# In[34]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[36]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[38]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[47]:


#Q3
df['floors'].value_counts().to_frame()


# In[61]:


#Q4
sns.boxplot(x='waterfront', y='price', data=df)


# In[63]:


#Q5
sns.regplot(x='sqft_above', y='price', data=df)


# In[65]:


df.corr()['price'].sort_values()


# In[67]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[70]:


#Q6
lm.fit(df[['sqft_living']], df[['price']])

lm.score(df[['sqft_living']], df[['price']])


# In[77]:


#Q7
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] 

Y= df[['price']]

lm.fit(df[features], df[['price']])
lm.score(df[features], df[['price']])


# In[82]:


#Q8
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)

pipe.fit(df[features], df[['price']])

pipe.score(df[features], df[['price']])


# In[84]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[86]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[91]:


#Q9
from sklearn.linear_model import Ridge
RR=Ridge(alpha=0.1)
RR.fit(df[features], df[['price']])
RR.score(df[features], df[['price']])


# In[94]:


#Q10
pr=PolynomialFeatures(degree=2)

x_train_pr= pr.fit_transform(x_train)
x_test_pr= pr.fit_transform(x_test)

RR=Ridge(alpha=0.1)
RR.fit(x_train_pr, y_train)
RR.score(x_test_pr, y_test)

