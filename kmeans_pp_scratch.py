import numpy as np
import matplotlib as plt


def euclidean_dist(x1,x2):
    """
    x1:points [(m,)]
    x2: datas [(n,m)]
    o/p:dist [(n,)]
    """
    return np.sqrt(np.sum((x1-x2)**2,axis=1))

class K_means:
    def __init__(self, centers:np.ndarray, n_clusters:int, n_iter=300):
        self.max_iter=n_iter
        if centers:
            self.centroids = centers
            self.ncluster= len(centers)
        elif n_clusters:
            self.ncluster=n_clusters
        else:
            raise ValueError()
            print('Need more arguments')
        
    
    def fit(self,x_train:np.ndarray):
        #if no centroids provided initialize using kmeans++ and initialize the centroids with proportional probability distance
        if not self.centroid:
            self.centroids=[np.random.choice(x_train)] 
            for i in range(self.ncluster-1):
                #point dist from nearest cluster
                c_dists = np.sum([euclidean_dist(centroid,x_train) for centroid in self.centroids], axis=0)
                c_dists /=np.mean(c_dists) # normalize
                new_centroid_id, = np.random.choice(range(len(x_train)), size=1, p=c_dists)
                self.centroids += [x_train[new_centroid_id]]
            
        iteration =0
        previous_centroids=None
        
        while np.not_equal(previous_centroids,self.centroids) and iteration<self.max_iter:
            sorted_points = [[] for _ in range(self.ncluster)]
            for x in x_train:
                dists = euclidean_dist(x, self.centroids)
                centroid_id = np.argmin(dists)
                sorted_points[centroid_id].append(x)
            # Push current centroids to previous
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                # Catch any np.nans, resulting from a centroid having no points
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
        
    def eval(self,x_test):
        f_centroids=[]
        f_centroid_ids=[]

        for x in x_test:
            f_dists=euclidean_dist(x,self.centroids)
            f_centroid_id=np.argmin(f_dists)
            f_centroids.append(self.centroids[f_centroid_id])
            f_centroid_ids.append(f_centroid_id)

        return np.array(f_centroids)