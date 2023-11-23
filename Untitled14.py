#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# In[8]:


filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df=pd.read_csv(filepath)


# In[9]:


df=df._get_numeric_data()
df.head()


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')
def DistributionPlot(Redfunction, Bluefunction, Redname, Bluename, Title ):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1= sns.kdeplot(Redfunction, color='r', label= Redname )
    ax2= sns.kdeplot(Bluefunction, color='b', label= Bluename, ax=ax1)
    
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()


# In[50]:


def PollyPlot(x_train, x_test, y_train, y_test, lr, polytransform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    xmax=max([x_train.values.max(), x_test.values.max()])
    xmin=min([x_train.values.max(), x_test.values.max()])
    
    x=np.arange(xmin, xmax, 0.1)
    
    plt.plot(x_train, y_train, 'ro', label='Training data')
    plt.plot(x_test, y_test, 'ro', label='Training data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1,1))), label= 'Predicted function')
    
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


# In[17]:


y_data= df['price']
x_data= df.drop('price', axis=1)


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size= 0.1, random_state=1)

print("Number of test samples :", x_test.shape[0])
print("Number of training samples:", x_train.shape[0])

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size= 0.4, random_state=0)

print("Number of test samples :", x_test1.shape[0])
print("Number of training samples:", x_train1.shape[0])
# In[26]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train[['horsepower']], y_train)
lr.score(x_train[['horsepower']], y_train)
lr.score(x_test[['horsepower']], y_test)


# In[32]:


from sklearn.model_selection import cross_val_score

R_sq_cross= cross_val_score(lr, x_data[['horsepower']], y_data, cv=4)

R_sq_cross
print("The mean of the folds are", R_sq_cross.mean(), "and the standard deviation is" , R_sq_cross.std())


# In[35]:


from sklearn.model_selection import cross_val_predict

yhate= cross_val_predict(lr, x_data[['horsepower']], y_data, cv=4)

yhate[199:201]


# In[37]:


lr=LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)


# In[47]:


yhat_train=lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train


# In[42]:


yhat_test= lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test


# In[51]:


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[73]:


from sklearn.preprocessing import PolynomialFeatures

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.45, random_state= 0 )



# In[74]:


pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr


# In[75]:


poly=LinearRegression()
poly.fit(x_train_pr, y_train)


# In[76]:


yhatu= poly.predict(x_train_pr)
yhatu


# In[78]:


PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr)


# In[81]:


poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)


# In[86]:


Rsq_test=[]
order =[1,2,3,4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsq_test.append(lr.score(x_test_pr, y_test))
    
plt.plot(order, Rsq_test)
plt.xlabel("Order")
plt.ylabel("R_squared")
plt.title("R-squared of test data")



# In[87]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])


# In[88]:


from sklearn.linear_model import Ridge


# In[89]:


RigeModel= Ridge(alpha=1)


# In[91]:


RigeModel.fit(x_train_pr, y_train)


# In[93]:


yhatr= RigeModel.predict(x_train_pr)

yhatr


# In[95]:


from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[97]:


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


# In[100]:


from sklearn.model_selection import GridSearchCV
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1


# In[102]:


RR= Ridge()
RR


# In[106]:


Grid1= GridSearchCV(RR, parameters1, cv=4)

Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)


# In[110]:


BestRR= Grid1.best_estimator_
BestRR


# In[111]:


BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

