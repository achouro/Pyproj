#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
connection= sqlite3.connect("INSTRUCTOR.db")
cursor_object= connection.cursor()

table= """ CREATE TABLE IF NOT EXISTS INSTRUCTOR (ID INTEGER PRIMARY KEY NOT NULL , FNAME VARCHAR(20), LNAME VARCHAR(20),
           CITY VARCHAR(20),CCODE VARCHAR(2)); """
cursor_object.execute(table)
print("Table is Ready")



# In[2]:


cursor_object.execute(""" INSERT INTO INSTRUCTOR VALUES (1, 'Rav', 'Ahuja', 'TORONTO', 'CA'), 
                                                        (2, 'Raul', 'Chong', 'Markham', 'CA'), 
                                                        (3, 'Hima', 'Vasudevan', 'Chicago', 'US'); """)


# In[3]:


statement = '''SELECT * FROM INSTRUCTOR'''
cursor_object.execute(statement)
output =cursor_object.fetchall()
print("Hola hoop")
for row in output:
    print(row)
    


# In[11]:


statement = '''SELECT * FROM INSTRUCTOR'''
cursor_object.execute(statement)
output =cursor_object.fetchmany(2)
print("Hola hoop")
for row in output:
    print(row)


# In[40]:


import pandas as pd

statement = '''SELECT * FROM INSTRUCTOR'''
cursor_object.execute(statement)
output =cursor_object.fetchall()
print("Hola hoop")
print(pd.DataFrame(output, columns=['ID', 'FNAME','LNAME', 'CITY', 'COUNTRY']))


# In[103]:


query_update = '''UPDATE INSTRUCTOR SET CITY="MOOSETOWN" WHERE LNAME="AHUJA";'''
cursor_object.execute(query_update)


# In[105]:


query_update = '''UPDATE INSTRUCTOR SET CITY="MOOSETOWN" WHERE LNAME="Ahuja";'''
cursor_object.execute(query_update)

statement = '''SELECT * FROM INSTRUCTOR;'''
cursor_object.execute(statement)

output =cursor_object.fetchall()

for row in output: 
    print(row)


# In[110]:


query_update = '''UPDATE INSTRUCTOR SET CITY="MOOSETOWN" WHERE LNAME="Ahuja";'''
cursor_object.execute(query_update)

statement = '''SELECT * FROM INSTRUCTOR;'''
cursor_object.execute(statement)

output =cursor_object.fetchall()
print("Hola hoop")

print(pd.DataFrame(output, columns=['ID', 'FNAME','LNAME', 'CITY', 'COUNTRY']))


# In[117]:


df = pd.read_sql_query("select * from instructor;", connection)

df



# In[118]:


df.LNAME[0]


# In[121]:


connection.close()


# In[ ]:


Rolla= pd.DataFrame(columns=['ID', 'FNAME','LNAME', 'CITY', 'COUNTRY'])

statement = '''SELECT * FROM INSTRUCTOR'''
cursor_object.execute(statement)
output =cursor_object.fetchall()
print("Hola hoop")

for i,row in enumerate(output):
   
    Rolla= Rolla._append({'ID':0, 'FNAME':1,'LNAME':2, 'CITY':3, 'COUNTRY':4}, ignore_index=True)
    print(i,Rolla)


# In[ ]:


Rolla= pd.DataFrame(columns=['ID', 'FNAME','LNAME', 'CITY', 'COUNTRY'])

statement = '''SELECT * FROM INSTRUCTOR'''
cursor_object.execute(statement)
output =cursor_object.fetchall()
print("Hola hoop")

for row in output:
    ID=output[0].integer
    FNAME=output[0:2]
    LNAME=output[0:2]
    CITY=output[0:2]
    COUNTRY=output[0:2]
    Rolla= Rolla._append({'ID':ID, 'FNAME':FNAME,'LNAME':LNAME, 'CITY':CITY, 'COUNTRY':COUNTRY}, ignore_index=True)
print(Rolla)

