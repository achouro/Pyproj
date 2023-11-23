#!/usr/bin/env python
# coding: utf-8

# In[64]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[11]:


table="<table><tr><td id='flight'>Flight No</td><td>Launch site</td> <td>Payload mass</td></tr><tr> <td>1</td><td><a href='https://en.wikipedia.org/wiki/Florida'>Florida<a></td><td>300 kg</td></tr><tr><td>2</td><td><a href='https://en.wikipedia.org/wiki/Texas'>Texas</a></td><td>94 kg</td></tr><tr><td>3</td><td><a href='https://en.wikipedia.org/wiki/Florida'>Florida<a> </td><td>80 kg</td></tr></table>"
bst=BeautifulSoup(table, "html.parser")

bst


# In[7]:


table_rows=bst.find_all("tr")
table_rows


# In[16]:


for i, row in enumerate(table_rows):
    print("row",i)
    cell=row.find_all("td")
    for j, cells in enumerate(table_rows):
        print("column", j, cell )


# In[10]:


table_columns=bst.find_all("td")
table_columns


# In[17]:


for i, row in enumerate(table_rows):
    print("row",i)
    cell=row.find_all("td")
    for j, cells in enumerate(table_rows):
        print("column", j, cell )


# In[32]:


url = "http://www.ibm.com"

data= requests.get(url).text

soup= BeautifulSoup(data, "html.parser")

lll= soup.find_all('a',href=True)



# In[39]:


for j in enumerate(lll):
    print( i, "th link")
for  link in lll:
    print(  link.get('href'))


# In[42]:


for link in soup.find_all('img'):
    print(link)
    print(link.get('src'))
    


# In[51]:


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/labs/datasets/HTMLColorCodes.html"


data= requests.get(url).text

soup= BeautifulSoup(data, "html.parser")

tab= soup.find('table')




# In[63]:


for rowo in tab.find_all('tr'):
    cols= rowo.find_all('td')
    print("{}  {}".format(cols[2].string, cols[3].string))


# In[128]:


url = "https://en.wikipedia.org/wiki/World_population"

data  = requests.get(url).text

soup = BeautifulSoup(data,"html.parser")

tables = soup.find_all('table')


len(tables)



# In[127]:


for index, table in enumerate(tables):
    if("Population by region (2020 estimates)" in str(table)):
        indexed_table = index
print(indexed_table)

        


# In[98]:


print(tables[indexed_table].prettify())


# In[103]:


import pandas as pd


# In[141]:


popu_data= pd.DataFrame(columns=["Region", "Density", "Population", "Most populous country", "Most populous city"])

for row in tables[indexed_table].tbody.find_all("tr"):
    col=row.find_all("td")
    if (col!= []):
        reg= col[0].text.strip()
        den= col[1].text.strip()
        pop=col[2].text.strip()
        mpco=col[3].text.strip()
        mpci=col[4].text.strip()
        
        popu_data=popu_data._append({"Region":reg, "Density":den, "Population":pop, "Most populous country":mpco, "Most populous city":mpci}, ignore_index= True)

popu_data







# In[157]:


url = "https://en.wikipedia.org/wiki/World_population"

data  = requests.get(url).text

soup = BeautifulSoup(data,"html.parser")

tables = soup.find_all('table')



# In[160]:


for index, table in enumerate(tables):
    if("Population by region (2020 estimates)" in str(table)):
        indexed_table = index
print(indexed_table)



# In[1]:


pip install html5lib


# In[5]:


from bs4 import BeautifulSoup
import requests
import pandas as pd

import html5lib as hh

url = "https://en.wikipedia.org/wiki/World_population"

data  = requests.get(url).text

soup = BeautifulSoup(data,"html.parser")

tables = soup.find_all('table')

for index, table in enumerate(tables):
    if("Population by region (2020 estimates)" in str(table)):
        indexed_table = index
print(indexed_table)





# In[19]:


pd.read_html(str(tables), flavor='bs4')[7]


# In[30]:


pd.read_html(url, match= "10 most densely populated countries",flavor='bs4')[0]


# In[ ]:




