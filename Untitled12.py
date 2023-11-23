#!/usr/bin/env python
# coding: utf-8

# In[1]:


import piplite
await piplite.install(['pandas'])
await piplite.install(['matplotlib'])


# In[3]:


import pandas as pd
import matplotlib.pylab as plt


# In[8]:


get_ipython().system('pip install pyodide.http')


# In[11]:


from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())


# In[15]:


filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[18]:


df = pd.read_csv(filename, names = headers)
df.head()


# In[20]:


import numpy as np


df.replace("?", np.nan, inplace = True)
df.head(5)


# In[26]:


missing=df.isnull()
missing.make.count()

missing.columns.tolist()


# In[28]:


for column in missing.columns.tolist() :
    print(missing[column].value_counts())


# In[38]:


avg_nl=df['normalized-losses'].astype('float64').mean(axis=0)
avg_nl


# In[41]:


df['normalized-losses'].replace(np.NaN, avg_nl, inplace= True)
df.head(10)


# In[55]:


df['normalized-losses'].value_counts().idxmax()


# In[66]:


df.dropna(subset=['price'], axis=0, inplace=True)

df.reset_index(drop=True, inplace= True)
df.count()


# In[68]:


df.dtypes


# In[72]:


df[["bore", "stroke","peak-rpm", "price"]] = df[["bore", "stroke","peak-rpm", "price"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df.dtypes


# In[85]:


df["horsepower"]=df["horsepower"].dropna().astype("int")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


# In[89]:


plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel('HP')
plt.pyplot.ylabel('Frequency')
plt.pyplot.legend('HP bins')


# In[100]:


bins=np.linspace(df["horsepower"].min(), df["horsepower"].max(), 4)
bins
hp_groups=['Low', 'Medium', 'High']


# In[106]:


df['horsepower_binned']=pd.cut(df['horsepower'], bins, labels=hp_groups, include_lowest=True)


df[['horsepower_binned', 'horsepower' ]].head(10)


# In[142]:


df['horsepower_binned'].value_counts()


# In[119]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.bar(hp_groups, df["horsepower_binned"].value_counts())
plt.pyplot.xlabel('HP')
plt.pyplot.ylabel('Frequency')
plt.pyplot.title('HP bins')


# In[122]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df["horsepower"], bins = 3)
plt.pyplot.xlabel('HP')
plt.pyplot.ylabel('Frequency')
plt.pyplot.title('HP bins')


# In[152]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)


# In[163]:


df.drop(['fuel-type-diesel'],['fuel-type-gas'], axis = 1, inplace=True )
df.head()

