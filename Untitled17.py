#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df=pd.read_csv(path)
df.head()


# In[21]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(5)


# In[26]:


msk=np.random.rand(len(df))<0.8
train= cdf[msk]
test= cdf[~msk]


# In[27]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[28]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

regr.coef_
regr.intercept_


# In[30]:


split= np.random.rand(len(df))<0.8
train= df[split]
test= df[~split]


# In[48]:


from sklearn import linear_model
reg = linear_model.LinearRegression()

x_train= np.asanyarray(train[['ENGINESIZE']])
y_train= np.asanyarray(train[['CO2EMISSIONS']])

reg.fit(x_train, y_train)

print("Y=",  reg.intercept_ , "+", "X *", reg.coef_)


# In[37]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(x_train, reg.coef_[0][0]*x_train + regr.intercept_[0], '-r')
plt.xlabel("Engines Size")
plt.ylabel("CO2 Emissions")


# In[50]:


from sklearn.metrics import r2_score

x_test = np.asanyarray(test[['ENGINESIZE']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

y_hat_test= reg.predict(x_test)

R_sq= r2_score(x_test, y_hat_test)

R_sq


# In[52]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()


# In[54]:


split= np.random.rand(len(cdf))<0.8
train= cdf[split]
test= cdf[~split]


# In[56]:


from sklearn import linear_model
reg = linear_model.LinearRegression()

x_tr= np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_tr= np.asanyarray(train[['CO2EMISSIONS']])

reg.fit(x_tr, y_tr)

print("Y=",  reg.intercept_ , "+", "X *", reg.coef_)


# In[65]:


from sklearn.metrics import r2_score

x_te= np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_te= np.asanyarray(test[['CO2EMISSIONS']])

y_hat_te= reg.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])

R_sququ= r2_score(x_te, y_hat_te)

reg.score(x_te, y_hat_te)


# In[67]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[69]:


y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])


# In[70]:


regr.score(x, y)

