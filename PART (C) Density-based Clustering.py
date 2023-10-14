#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv("Mall_Customers.csv")


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.shape


# In[12]:


y = df.iloc[:, [3,4]].values


# In[14]:


y


# In[15]:


plt.scatter(y[:,0], y[:,1], s=10, c= "black")


# In[16]:


dbscan = DBSCAN(eps=5, min_samples=5)


# In[17]:


labels = dbscan.fit_predict(y)


# In[18]:


np.unique(labels)


# In[20]:


cluster_df = pd.DataFrame(labels, columns=['Cluster'])
df = pd.concat([df, cluster_df], axis=1)
print(df)
# Visualising the clusters
plt.scatter(y[labels == -1, 0], y[labels == -1, 1], s = 10, c = 'black') 

plt.scatter(y[labels == 0, 0], y[labels == 0, 1], s = 10, c = 'blue')
plt.scatter(y[labels == 1, 0], y[labels == 1, 1], s = 10, c = 'red')
plt.scatter(y[labels == 2, 0], y[labels == 2, 1], s = 10, c = 'green')
plt.scatter(y[labels == 3, 0], y[labels == 3, 1], s = 10, c = 'brown')
plt.scatter(y[labels == 4, 0], y[labels == 4, 1], s = 10, c = 'pink')
plt.scatter(y[labels == 5, 0], y[labels == 5, 1], s = 10, c = 'yellow')      
plt.scatter(y[labels == 6, 0], y[labels == 6, 1], s = 10, c = 'silver')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:




