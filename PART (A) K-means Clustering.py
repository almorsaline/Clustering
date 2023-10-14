#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
# Assuming you have a CSV file named 'mall_customer.csv' in the current directory
file_path = 'Mall_Customers.csv'  # Update with the actual path to your file
mall_data = pd.read_csv(file_path)
print(mall_data)

# Convert 'Gender' to binary (1 for Male, 0 for Female)
mall_data['Genre'] = mall_data['Genre'].apply(lambda x: 1 if x == 'Male' else 0)

# Extract relevant features for clustering
X = mall_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre']].values

# Iterate through the range of K values (1 to 15) and calculate SSE
k_values = range(1, 16)  # K values from 1 to 15
sse = []  # to store SSE for each K

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot the SSE for each K
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method to find the optimal K')
plt.xticks(range(1, 16))
plt.show()


# In[5]:


# Using the optimal K to perform K-means clustering
optimal_k = 5  # Replace with the optimal K you obtained

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)  # X is your feature matrix


# In[7]:


# Get the cluster assignments for each data point
cluster_assignments = kmeans.labels_
mall_data['Cluster'] = cluster_assignments
print(mall_data)


# In[10]:


import matplotlib.pyplot as plt

# Scatter plot for visualization of clusters
plt.figure(figsize=(10, 6))

# Color map for clusters
colors = plt.cm.get_cmap('tab20', optimal_k)

for i in range(optimal_k):
    plt.scatter(X[cluster_assignments == i, 0], X[cluster_assignments == i, 1], s=50, c=colors(i), label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of Customers')
plt.legend()
plt.show()


# In[ ]:




