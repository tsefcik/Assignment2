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
            print(centroids)

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

        print(centroids)
        print(cluster_list)
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
    def silhouette_coefficient(self, cluster):
        total_dist_0 = 0
        total_dist_1 = 0
        count_0 = 0
        count_1 = 0
        for point in range(0, len(cluster)):
            for other_point in range(len(cluster)):
                if point == 0 and other_point == 0:
                    current_point = self.data
                    other_point = self.data.loc[other_point].tolist()

                    dist = self.euclidean_distance(point1=current_point, point2=other_point)
                    total_dist_0 = total_dist_0 + dist
                    count_0 = count_0 + 1
                elif point == 1 and other_point == 1:
                    current_point = self.data.loc[point].tolist()
                    other_point = self.data.loc[other_point].tolist()

                    dist = self.euclidean_distance(point1=current_point, point2=other_point)
                    total_dist_1 = total_dist_1 + dist
                    count_1 = count_1 + 1

        average_dist_0 = total_dist_0 / count_0
        average_dist_1 = total_dist_1 / count_1

        print(average_dist_0)
        print(average_dist_1)

        return average_dist_0, average_dist_1
