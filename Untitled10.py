#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('load_ext', 'sql')


# In[6]:


import csv, sqlite3


# In[3]:


get_ipython().system('pip install -q pandas==1.1.5')


# In[8]:


connection= sqlite3.connect("socioeconomic.db")
cursor= connection.cursor()


# In[9]:


get_ipython().run_line_magic('sql', 'sqlite:///socioeconomic.db')
    


# In[12]:


import pandas as pd
url= 'https://data.cityofchicago.org/resource/jcxq-k9xf.csv'
df= pd.read_csv(url)
df.to_sql("chicago_socioeconomic_data", connection, if_exists='replace', index=False,method="multi")


# In[14]:


get_ipython().run_line_magic('sql', 'SELECT * FROM chicago_socioeconomic_data LIMIT 10;')


# In[16]:


con=sqlite3.connect("RealWorldData.db")
cur=con.cursor()


# In[ ]:




