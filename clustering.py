import pandas as pd
pd.set_option('display.max_columns', None)

# Load the excel file into a DataFrame
df = pd.read_excel('Footystats_Data_Science_Demo.xlsx')

# Display the first few rows to ensure it's loaded correctly
print(df.head())


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Extract the 'Asian Handicap' column for clustering
X = df[['Asian Handicap']]

# Using KMeans for clustering (let's take 2 clusters as an example)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Adding the cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Plotting the clusters
plt.scatter(df['Date'], df['Asian Handicap'], c=df['Cluster'])
plt.title('K-means Clustering on Asian Handicap')
plt.xlabel('Date')
plt.ylabel('Asian Handicap')
plt.show()

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Using AgglomerativeClustering with complete linkage
agg_clustering = AgglomerativeClustering(linkage='complete', n_clusters=2)
df['Agg_Cluster'] = agg_clustering.fit_predict(X)

# Plotting Dendrogram
plt.figure(figsize=(15, 7))
dendrogram = sch.dendrogram(sch.linkage(X, method='complete'))
plt.title('Dendrogram using Complete Linkage')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distances')
plt.show()

# Scatter plot of the clusters
plt.scatter(df['Date'], df['Asian Handicap'], c=df['Agg_Cluster'])
plt.title('Agglomerative Clustering on Asian Handicap')
plt.xlabel('Date')
plt.ylabel('Asian Handicap')
plt.show()

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

data = X['Asian Handicap'].values

# Make sure data is 2D
alldata = np.vstack((data)).T  # Transpose to make it 2D

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    alldata, 2, 2, error=0.005, maxiter=1000, init=None)

# Store fpc values for later plots
fpcs = []
fpcs.append(fpc)

# Plot clustered data
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model')

for j in range(2):
    ax2.plot(data, u[j, :], 'o', label='series ' + str(j))
    print(data.shape)
    print(u[j, :].shape)

ax2.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error


# Features and target for regression
X_regress = df[['LAway', 'Exponential', 'Coefficient']]
y_regress = df['Average']

# Split the dataset for regression
X_train_regress, X_test_regress, y_train_regress, y_test_regress = train_test_split(X_regress, y_regress, test_size=0.2, random_state=42)

# KNN for Regression
knn_regressor = KNeighborsRegressor(n_neighbors=5, metric='euclidean')  # Adjust 'n_neighbors' and 'metric' as needed
knn_regressor.fit(X_train_regress, y_train_regress)
y_pred_regress = knn_regressor.predict(X_test_regress)

mse = mean_squared_error(y_test_regress, y_pred_regress)
print('..............')

print(f"Mean Squared Error for Regression: {mse}")
print('..............')

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Reshape the Asian Handicap data to fit the algorithm requirements
data = df['Asian Handicap'].values.reshape(-1, 1)

# Apply DBSCAN clustering
# Note: eps and min_samples are critical parameters to be tuned for the DBSCAN algorithm
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(data)

# Get cluster labels and number of clusters (ignoring noise if present)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # '-1' labels are considered as noise by DBSCAN

# Add the cluster labels to the dataframe
df['DBSCAN_Cluster'] = labels

# Plot the results
plt.scatter(df['Date'], df['Asian Handicap'], c=labels, cmap='rainbow')
plt.title(f'DBSCAN Clustering (Estimated number of clusters: {n_clusters})')
plt.xlabel('Date')
plt.ylabel('Asian Handicap')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Define predictors and target
X = df[['Asian Handicap']]
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using KNN with Euclidean distance
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_euclidean.fit(X_train, y_train)
y_pred_euclidean = knn_euclidean.predict(X_test)

print("Classification report for KNN with Euclidean distance:\n")
print(classification_report(y_test, y_pred_euclidean))

# Using KNN with Manhattan distance
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan.fit(X_train, y_train)
y_pred_manhattan = knn_manhattan.predict(X_test)

print("\nClassification report for KNN with Manhattan distance:\n")
print(classification_report(y_test, y_pred_manhattan))


# ... [previous imports and data splitting]

# Using KNN with Euclidean distance
knn_euclidean.fit(X_train, y_train)
y_pred_euclidean = knn_euclidean.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Asian Handicap'], y_test, color='blue', label='True Class')
plt.scatter(X_test['Asian Handicap'], y_pred_euclidean, color='red', marker='x', label='Predicted Class')
plt.title('KNN Classification with Euclidean Distance')
plt.xlabel('Asian Handicap')
plt.ylabel('Class')
plt.legend()
plt.show()

# Using KNN with Manhattan distance
knn_manhattan.fit(X_train, y_train)
y_pred_manhattan = knn_manhattan.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Asian Handicap'], y_test, color='blue', label='True Class')
plt.scatter(X_test['Asian Handicap'], y_pred_manhattan, color='red', marker='x', label='Predicted Class')
plt.title('KNN Classification with Manhattan Distance')
plt.xlabel('Asian Handicap')
plt.ylabel('Class')
plt.legend()
plt.show()



