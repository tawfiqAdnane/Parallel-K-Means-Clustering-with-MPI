import numpy as np
from mpi4py import MPI
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# Place K centroids at random locations

def random_centroids(x, K):    
   n,m = np.shape(x)

   cent = x[random.sample(range(0, len(x)), K)]

   return cent

# Assigner tous les points de données au centroïde le plus proche
def assign_cluster(X, centroids):
    m = X.shape[0]
    clusters = np.zeros(m, dtype=int)
    for i in range(m):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        clusters[i] = np.argmin(distances)
    return clusters

# Calculer les nouveaux centroids comme la moyenne de tous les points du cluster
def new_centroids(X, clusters, K):
    new_centroids = np.array([X[clusters==i].mean(axis=0) for i in range(K)])
    return new_centroids

def kmeans_clustering(all_vals,K,max_iter = 100 ):
    centroids = random_centroids(X, K)
    for _ in range(max_iter):
        prev_centroids = centroids.copy()
        clusters = assign_cluster(X, centroids)
        centroids = new_centroids(X, clusters, K)
        if np.all(prev_centroids == centroids):
            break
    return centroids, clusters

#load the iris dataset
iris = load_iris()
#create features and target arrays
X = iris.data
y = iris.target

#split X, training and test data

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.3,random_state = 42,shuffle = True)

start_time = time.time()
centroids, clusters = kmeans_clustering(X[:,2:],3,max_iter = 100 )
centroids1, clusters1 = kmeans_clustering(X[:,:2],3,max_iter = 100 )
end_time = time.time()
print('Temps d\'exécution: ', end_time - start_time)


def kmeans_parallel(X, K, max_iters=100):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('Number of Processus:', size)
    # Scatter the data to all processes
    local_data = None
    if rank == 0:
        m, n = X.shape
        local_data = np.array_split(X, size)
    local_data = comm.scatter(local_data, root=0)
    # Randomly initialize centroids in the root process
    centroids = None
    if rank == 0:
        centroids = random_centroids(X, K)
    centroids = comm.bcast(centroids, root=0)

    for _ in range(max_iters):
        # Assign clusters to data points in each process
        clusters = assign_cluster(local_data, centroids)

        # Gather all clusters from different processes to the root process
        all_clusters = comm.gather(clusters, root=0)

        if rank == 0:
            # Concatenate the clusters from all processes
            all_clusters = np.concatenate(all_clusters)

            # Update centroids in the root process
            centroids = new_centroids(X, all_clusters, K)

        # Broadcast the updated centroids to all processes
        centroids = comm.bcast(centroids, root=0)

    # Gather the final centroids from all processes to the root process
    final_centroids = comm.gather(centroids, root=0)

    # Gather the final clusters from all processes to the root process
    final_clusters = comm.gather(clusters, root=0)

    if rank == 0:
        # Concatenate the centroids from all processes
        final_centroids = np.concatenate(final_centroids)

        # Concatenate the clusters from all processes
        final_clusters = np.concatenate(final_clusters)

        return final_centroids, final_clusters
    else:
        return None, None


import time

start_time = time.time()
final_centroids, final_clusters = kmeans_parallel(X,3,max_iters = 100)
end_time = time.time()
print('Temps d\'exécution2: ', end_time - start_time)

plt.scatter(X[:,2],X[:,3], c=final_clusters,)
plt.scatter(final_centroids[:,0],final_centroids[:,1], c="r", s=140, marker= "*")
plt.title('petal Clustering')
plt.show()



