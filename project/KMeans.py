import numpy as np


class KMeans:

    """
    Setup KMeans by giving it the value of k (num_classes), and a dataset.
    """
    def __init__(self, num_classes, data):
        # Number of clusters
        self.k = num_classes
        # Dataset that matches what SFS told us to use
        self.data = data

    """
    Algorithm to run kmeans on the given dataset with the number of classes in the data equal to the number
    of clusters we will use.
    """
    def kmeans_alg(self):
        unique_identifier = True

        while unique_identifier:
            # Setup random centroids from the given data
            centroids = self.data.sample(n=self.k).values

            if np.array_equal(centroids[0], centroids[1]):
                unique_identifier = True
            else:
                unique_identifier = False

        # Set the initial distance between centroids to be a very large number
        dist_centroids = np.Infinity
        # Initialize cluster list
        cluster_list = [0]*len(self.data)

        while True:
            for index, row in self.data.iterrows():
                # Get distance between the row and centroids
                dist = self.euclidean_distance(point1=self.data.loc[index].tolist(), point2=centroids[0])
                dist2 = self.euclidean_distance(point1=self.data.loc[index].tolist(), point2=centroids[1])

                # Assign cluster by assigning data point to closest centroid
                if dist < dist2:
                    cluster = 0
                    cluster_list.insert(index, cluster)
                    cluster_list.pop(len(cluster_list)-1)
                else:
                    cluster = 1
                    cluster_list.insert(index, cluster)
                    cluster_list.pop(len(cluster_list)-1)

            # Need to keep to determine if new centroids have moved
            old_centroids = centroids

            # Iterate through centroids and update with mean of row
            for index in range(0, len(centroids)):
                centroid_mean = np.mean(centroids[index])
                centroids[index] = centroid_mean

            # Get distance for old centroids and new one to compare if it has moved at all
            old_new_centroid_dist = self.euclidean_distance(point1=centroids, point2=old_centroids)
            if old_new_centroid_dist == dist_centroids:
                break
            else:
                dist_centroids = old_new_centroid_dist

        return centroids, cluster_list

    """
    Returns the Euclidean distance between two data points.
    Uses numpy to handle the complicated math.
    """
    def euclidean_distance(self, point1, point2):
        distance = np.sqrt(np.sum((point1-point2))**2)
        return distance

    """
    Returns the silhouette coefficient for clusters.
    """
    def silhouette_coefficient(self, cluster_list):
        silhouette_coefficient = 0
        for cluster_type in range(0, self.k):
            cluster_dist = 0
            cluster_dist_to_other = 0
            for point in range(0, len(cluster_list)):
                if cluster_list[point] == cluster_type:
                    cluster_dist = cluster_dist + self.euclidean_distance(point1=self.data.loc[cluster_type],
                                                                          point2=self.data.loc[point])
                else:
                    cluster_dist_to_other = cluster_dist_to_other + self.euclidean_distance(point1=self.data.loc[cluster_type],
                                                                          point2=self.data.loc[point])
            average_cluster_dist = np.mean(cluster_dist)
            average_cluster_dist_to_other = np.mean(cluster_dist_to_other)

            silhouette_coefficient = silhouette_coefficient + ((average_cluster_dist_to_other - average_cluster_dist) /
                                                               max(average_cluster_dist, average_cluster_dist_to_other))
        mean_silhouette_coefficient = np.mean(silhouette_coefficient)

        return mean_silhouette_coefficient

