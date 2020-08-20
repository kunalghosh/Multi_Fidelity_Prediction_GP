import sklearn.cluster
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

def cluster(heldout_set: list, n_clusters: int, random_seed, mbtr_data) -> (list, list):
  """
  first list returned is a list of indices indicating which cluster each molecule belongs to.
  second one is the cluster centers
  """
  # TODO: load mbtr data from the dataclass
  # mbtr_data = dataclass.mbtr_data
  
  X_train = mbtr_data[heldout_set, :]
  
  #-- Preprocessing                                       
  X_train = X_train.toarray()  
  
  #-- Clustering
  km = sklearn.cluster.KMeans(n_clusters = n_clusters, n_jobs = 4, random_state=random_seed)
  z_km = km.fit(X_train)
  
  # return (cluster_assignment, cluster_centers)
  labels = np.array(z_km.labels_)
  centers = np.array(z_km.cluster_centers_)
  
  
  return labels, centers
      	
def get_closest_to_center(heldout_set: list, cluster_assignment: list, cluster_centers: list, mbtr_data) -> list:
  #do we need labesl?
  """
  Molecules in the cluster which are closest to the cluster center.
  """
  #TODO: load mbtr data from the dataclass
  # mbtr_data = dataclass.mbtr_data
  
  X_train = mbtr_data[heldout_set, :]
  
  #-- Preprocessing                                       
  X_train_pp = X_train.toarray() 
  
  closest, _ = pairwise_distances_argmin_min(cluster_centers, X_train)
  
  return closest
