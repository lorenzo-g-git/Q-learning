#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=[
    [1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,7,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,28,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,84,0,0,0,0,0,0,0,0,0],
    [0,0,0,35,175,0,0,0,0,0,0,0,0],
    [0,0,21,0,140,294,0,0,0,0,0,0,0],
    [0,7,0,105,0,350,413,0,0,0,0,0,0],
    [1,0,42,0,315,0,700,462,0,0,0,0,0],
    [0,7,0,147,0,735,35,1050,41,0,0,0,0],
    [0,0,28,0,392,105,1260,140,1260,350,0,0,0],
    [0,0,0,84,126,735,420,1680,350,1190,210,0,0],
    [0,0,0,77,168,525,1050,1050,1680,560,840,105,0],
    [0,0,28,0,350,252,1365,1085,1680,1295,630,420,35],
    [0,7,0,147,0,980,266,2310,945,1890,770,420,140],
    [1,0,42,0,462,0,1820,420,2730,840,1260,420,140],
    [0,7,0,147,0,980,266,2310,945,1890,770,420,140],
    [0,0,28,0,350,252,1365,1085,1680,1295,630,420,35],
    [0,0,0,77,168,525,1050,1050,1680,560,840,105,0],
    [0,0,0,84,126,735,420,1680,350,1190,210,0,0],
    [0,0,28,0,392,105,1260,140,1260,350,0,0,0],
    [0,7,0,147,0,735,35,1050,41,0,0,0,0],
    [1,0,42,0,315,0,700,462,0,0,0,0,0],
    [0,7,0,105,0,350,413,0,0,0,0,0,0],
    [0,0,21,0,140,294,0,0,0,0,0,0,0],
    [0,0,0,35,175,0,0,0,0,0,0,0,0],
    [0,0,0,84,0,0,0,0,0,0,0,0,0],
    [0,0,28,0,0,0,0,0,0,0,0,0,0],
    [0,7,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0],
]


# In[3]:


import matplotlib.pyplot as plt

fig_1, ax =plt.subplots(1,1)
ax.axis('tight')
ax.axis('off')
the_table=ax.table(cellText=data,loc="center")

plt.show()


# In[4]:


import Q_learn


# In[10]:


from Q_learn import get_shortest_path
from Q_learn import get_array
from Q_learn import drow_path


# In[6]:


V=get_shortest_path(5,2)


# In[9]:


my_array=get_array(V)


# In[11]:


drow_path(the_table,my_array)


# In[12]:


fig_1


# In[14]:


V=get_shortest_path(7,6)


# In[ ]:




