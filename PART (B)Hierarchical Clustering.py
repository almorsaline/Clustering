#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
# Assuming the CSV file is in the same directory as this script
file_path = 'Mall_Customers.csv'  # Update with the actual path to your file
mall_data = pd.read_csv(file_path)

# Convert 'Genre' to binary (1 for Male, 0 for Female)
mall_data['Genre'] = mall_data['Genre'].apply(lambda x: 1 if x == 'Male' else 0)

# Prepare the features for clustering
X = mall_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre']].values

# Determine the number of clusters (optional)
# Generate the linkage matrix
linked = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()



# In[3]:


# Add the 'Cluster' column to mall_data
mall_data['Cluster'] = cluster_labels

# Display the updated DataFrame with the cluster labels
print(mall_data.head())

# Fit hierarchical clustering model
# Assuming we decide to have 3 clusters based on the dendrogram
n_clusters = 3

hierarchical = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
cluster_labels = hierarchical.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 6))

for i in range(n_clusters):
    plt.scatter(X[cluster_labels == i, 1], X[cluster_labels == i, 2], s=50, label=f'Cluster {i+1}')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clustering')
plt.legend()
plt.show()


# In[ ]:




