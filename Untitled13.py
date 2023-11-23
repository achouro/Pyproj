#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df=pd.read_csv(filepath)


# In[5]:


from sklearn.linear_model import LinearRegression


# In[7]:


lm=LinearRegression()
lm


# In[23]:


X= df[['highway-mpg']]
Y= df['price']

lm.fit(X,Y)

Yhat= lm.predict(X)

print("Y=", lm.intercept_, "+", lm.coef_,"*X")


# In[28]:


lm1=LinearRegression()

X= df[['engine-size']]
Y= df['price']
lm1.fit(X, Y)
Yhat=lm1.predict(X)
print("Price=", lm1.intercept_, "+", lm1.coef_,"*EngineSize")


# In[59]:


Z= df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y= df['price']
lm.fit(Z,Y)
Yhat=lm.predict(Z)
print("Price=", lm.intercept_, "+", lm.coef_[0],"HorsePower", "+", 
      lm.coef_[1],"CurbWeight", "+", lm.coef_[2],"EngineSize", "+", lm.coef_[3],"HighwayMPG", )


# In[40]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


width= 8
height= 5
plt.figure(figsize=(width, height))
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)


# In[55]:


plt.figure(figsize=(width, height))
sns.residplot(x= 'highway-mpg', y='price', data=df)


# In[64]:


Y= df['price']
Yhat= lm.predict(Z)

plt.figure(figsize=(width, height))

ax1= sns.distplot(Y, hist=False, color= 'r', label='Actual Value')
sns.distplot(Yhat, hist=False, color= 'b', label='Fitted Value', ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# In[80]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[76]:


x=df['highway-mpg']
y=df['price']


# In[77]:


fit=np.polyfit(x,y,3)
display= np.poly1d(fit)
print(display)


# In[84]:


PlotPolly(display, x, y , 'Bobby Gas Miles/Gallon')


# In[88]:


x=df['highway-mpg']
y=df['price']
fit1=np.polyfit(x,y, 11)
display1=np.poly1d(fit1)
print(display1)
PlotPolly(display1, x, y , 'Bobby Gas Miles/Gallon')


# In[101]:


from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)

Z_pr= pr.fit_transform(Z)
Z_pr.shape


# In[103]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[107]:


Input= [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('Model', LinearRegression())]

pipe=Pipeline(Input)
pipe


# In[112]:


Z=Z.astype(float)
pipe.fit(Z, Y)
ypipe= pipe.predict(Z)



# In[135]:


plt.plot(Y-ypipe )


# In[140]:


lm.fit(X, Y)

print('The R-square is: ', lm.score(X, Y))
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


# In[134]:


from sklearn.metrics import mean_squared_error

mse= mean_squared_error(Y, Yhat)
print('The mean square error of price and predicted value is: ', mse)


# In[143]:


from sklearn.metrics import r2_score
rsq= r2_score(Y, display(x))
rsq
mse= mean_squared_error(Y, Yhat)
mse


# In[155]:


new_input= np.arange(1,100, 1).reshape(-1,1)

lm.fit(X,Y)

yhat=lm.predict(new_input)
yhat[0:5]



# In[152]:


plt.plot(new_input, yhat)

