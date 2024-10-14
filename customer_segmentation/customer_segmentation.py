import pandas as pd

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Display the first few rows of the dataset
print(data.head())

print(data.isnull().sum())

features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K_range = range(1, 11)  # Testing for 1 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


kmeans = KMeans(n_clusters=4)  # Replace 4 with your chosen number
kmeans.fit(features)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.colorbar(label='Cluster')
plt.show()


