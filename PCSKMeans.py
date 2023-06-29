import numpy as np
from scipy import spatial as sdist
import copy
from sklearn.metrics import pairwise_distances
import time

def l2_distance(point1, point2):
	return sum([(float(i) - float(j)) ** 2 for (i, j) in zip(point1, point2)])

def tolerance(tol, dataset):
	n = len(dataset)
	dim = len(dataset[0])
	averages = [sum(dataset[i][d] for i in range(n)) / float(n) for d in range(dim)]
	variances = [sum((dataset[i][d] - averages[d]) ** 2 for i in range(n)) / float(n) for d in range(dim)]
	return tol * sum(variances) / dim

def binary_search_delta(obj, s, max_iter = 200):

	#Norma de Frobenius por defecto
	frob_norm = np.linalg.norm(obj)

	if frob_norm == 0 or np.sum(np.abs(obj/frob_norm)) <= s:

		return 0

	else:

		lam1 = 0
		lam2 = np.max(np.abs(obj)) - 1e-5
		iters = 0

		while iters < max_iter and (lam2 - lam1) > 1e-4:

			su = np.sign(obj) * (np.abs(obj)-((lam1+lam2)/2)).clip(min = 0)

			if np.sum(abs(su/np.linalg.norm(su,2))) < s:

				lam2 = (lam1+lam2)/2

			else:

				lam1 = (lam1+lam2)/2

			iters += 1

		return (lam1+lam2)/2


def PCSKMeans(x, k, const_mat, sparsity = 1.1, tol=1e-4, max_iter = 20000, init_centroids = []):

	#n = number of instances / d = number of features
	n, d = np.shape(x)

	tol = tolerance(tol, x)
	
	#Initialize centroids
	if len(init_centroids) == 0:

		centroids = np.random.rand(k, d)
		dataset_diameter = np.max(pairwise_distances(x, metric='euclidean'))

		for i in range(k):
			centroids[i, :] = centroids[i, :] + np.min(x, 0)
	else:
		centroids = init_centroids

	start_time = time.time()
	#Get maximally separated instances
	max_distance = np.power(np.max(x, axis = 0) - np.min(x, axis = 0), 2)

	#Initialize weights
	w = np.ones(d, dtype = np.float) * np.sqrt(d)

	#Compute global centroid
	global_centroid = np.mean(x, axis = 0)
	distance_to_global_centroid = np.sum(np.power(x - global_centroid, 2), axis = 0)

	#Initialize partition
	partition = np.ones(n) * -1

	
	iters = 0
	shift = -1

	while iters < max_iter:

		iters += 1

		#Assign each instance to its closest cluster centroid
		for i in range(n):

			instance = x[i,:]

			#Compute weigthed squared euclidean distances
			weighted_squared_diffs = np.sum(np.power(centroids - instance, 2) * w, axis = 1)

			#Compute penalties
			penalties = np.zeros(k)

			for l in range(k):

				for j in range(n):

					#if the second instance has a label
					if partition[j] != -1:

						#if ML(i,j) and instance i is going to be assigned to a label other than the label of instance j
						if const_mat[i,j] == 1 and l != partition[j]:

							penalties[l] += np.sum(np.power(instance - x[j,:], 2) * w)

						#if ML(i,j) and instance i is going to be assigned to a label equal to the label of instance j
						if const_mat[i,j] == -1 and l == partition[j]:

							penalties[l] += np.sum((max_distance - np.power(instance - x[j,:], 2)) * w)

			partition[i] = np.argmin(weighted_squared_diffs + penalties)

		#Recompute centroids
		old_centroids = copy.deepcopy(centroids)
		for i in range(k):

			cluster = x[np.where(partition == i)[0],:]
			if cluster.shape[0] > 0:
				centroids[i,:] = np.mean(cluster, axis = 0)

		#Update weights
		within_cluster_distances = np.zeros(d)
		gammas = np.zeros(d)

		for i in range(d):

			for l in range(k):

				cluster_indices = np.where(partition == l)[0]
				cluster = x[cluster_indices,:]

				within_cluster_distance = np.power(cluster[:,i] - centroids[l,i], 2)

				#Compute penalties
				penalties = np.zeros(len(cluster_indices))

				for j in range(len(cluster_indices)):

					instance_index = cluster_indices[j]

					for m in range(n):

						if const_mat[instance_index,m] == 1 and partition[instance_index] != partition[m]:

							penalties[j] += ((x[instance_index,i] - x[m,i])**2)

						if const_mat[instance_index,m] == -1 and partition[instance_index] == partition[m]:

							penalties[j] += (max_distance[i] - (x[instance_index,i] - x[m,i])**2)
			
			within_cluster_distances[i] = np.sum(within_cluster_distance)
			gammas[i] = distance_to_global_centroid[i] - np.sum(within_cluster_distance) + np.sum(penalties)

		delta = binary_search_delta(distance_to_global_centroid - within_cluster_distances, sparsity)

		w = (np.sign(gammas) * (np.abs(gammas)- delta).clip(min = 0)) / np.linalg.norm(np.sign(gammas) * (np.abs(gammas) - delta))

		#Compute centroid shift for stopping criteria
		shift = sum(l2_distance(centroids[i], old_centroids[i]) for i in range(k))

		if shift <= tol:
			break

	return partition, w, iters, time.time() - start_time