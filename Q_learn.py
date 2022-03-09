#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import statistics


# In[6]:


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


# In[7]:


import matplotlib.pyplot as plt

fig_1, ax =plt.subplots(1,1)
ax.axis('tight')
ax.axis('off')
the_table=ax.table(cellText=data,loc="center")

plt.show()


# In[8]:


## definisco un algoritmo di minimo costo per ragiungere la mia meta
## definiamo la struttura del nostro ambiente
environment_rows = 29
environment_columns = 13
## ora per ogni stato avremo una tupla (s,a), s=(E,S)
q_values = np.zeros((environment_rows, environment_columns, 8))
# definiamo le azioni: sono 8, su, giù, destra, sinistra e le oblique
actions = ['up', 'right', 'down', 'left','upright','upleft','downright','downleft']


# In[9]:


rewards = np.array(data)
rewards= rewards.reshape(environment_rows, environment_columns)
rewards=rewards* -1
rewards[rewards == 0 ] = -10000
rewards[7][0] = 100
rewards


# In[10]:


## Trainiamo il nostro modello

## cominciamo definendo le funzioni di scelta
#define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
  if rewards[current_row_index, current_column_index]!= -10000. and rewards[current_row_index, current_column_index]!= 100.:
    return False
  else:
    return True



# In[11]:


def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index
  


# In[12]:


def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else: #choose a random action
        return np.random.randint(8)


# In[13]:


def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    elif actions[action_index] == 'upright' and current_row_index > 0 and current_column_index < environment_columns - 1:
        new_row_index -= 1
        new_column_index +=1
    elif actions[action_index] == 'upleft'  and current_row_index > 0 and current_column_index > 0:
        new_row_index -= 1
        new_column_index -=1        
    elif actions[action_index] == 'downright' and current_row_index < environment_rows - 1 and current_column_index < environment_columns - 1:
        new_row_index += 1
        new_column_index +=1 
    elif actions[action_index] == 'downleft'  and current_row_index < environment_rows - 1 and current_column_index > 0:
        new_row_index += 1
        new_column_index -=1 
    return new_row_index, new_column_index
        


# In[14]:


def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
         return []
    else: #if this is a 'legal' starting location
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
            print(shortest_path)
        return shortest_path
        


# In[15]:


epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the agent should learn
for episode in range(1000):
    row_index, column_index = get_starting_location()
    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')      


# In[25]:


def get_array(l):
    my_array=np.array(l)
    
    return my_array


# In[18]:


def drow_path(table,v):
    h=v.shape
    h=np.array(h)
    k=h[0]
    for i in range(h[0]):
         table[(v[i][0],v[i][1])].set_facecolor("#b70210")
    return table 


# In[ ]:




