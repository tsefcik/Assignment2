from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Spambase as spambase
from project import SFS as sfs
from project import KMeans as km
import sys


# SFS for Iris
def run_sfs_iris(filename, target_class):
    # Setup data
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris(filename=filename, target_class=target_class)
    # Split the data set into 2/3 and 1/3
    iris_data_train = iris_data.sample(frac=.667)
    iris_data_test = iris_data.drop(iris_data_train.index)

    # Setup SFS object
    stepwise_forward = sfs.SFS()

    # Setup SFS features to go through
    feature_set_columns = list()
    for feature in iris_data.columns:
        feature_set_columns.append(feature)
    feature_set_columns.remove(target_class)

    # Separate data for taining/testing purposes
    iris_data_train_target_class = iris_data_train[target_class]
    iris_data_train_features = iris_data_train.iloc[:, 0:4]
    iris_data_test_features = iris_data_test.iloc[:, 0:4]

    # Run SFS
    best_features = stepwise_forward.perform_sfs(feature_set_columns=feature_set_columns,
                                                 d_train=iris_data_train_features,
                                                 d_valid=iris_data_test_features,
                                                 predicted=iris_data_train_target_class)
    # Return dataset with best features only
    best_data = iris_data[best_features]

    return best_features, best_data


# SFS for Glass
def run_sfs_glass(filename, target_class):
    # Setup data
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass(filename=filename, target_class=target_class)

    # Split the data set into 2/3 and 1/3
    glass_data_train = glass_data.sample(frac=.667)
    glass_data_test = glass_data.drop(glass_data_train.index)

    # Setup SFS object
    stepwise_forward = sfs.SFS()

    feature_set_columns = list()
    for feature in glass_data.columns:
        feature_set_columns.append(feature)
    feature_set_columns.remove(target_class)

    glass_data_train_target_class = glass_data_train[target_class]
    glass_data_train_features = glass_data_train.iloc[:, 0:9]
    glass_data_test_features = glass_data_test.iloc[:, 0:9]

    best_features = stepwise_forward.perform_sfs(feature_set_columns=feature_set_columns,
                                                 d_train=glass_data_train_features,
                                                 d_valid=glass_data_test_features,
                                                 predicted=glass_data_train_target_class)

    # Return dataset with best features only
    best_data = glass_data[best_features]

    return best_features, best_data


# SFS for Spambase
def run_sfs_spambase(filename, target_class):
    # Setup data
    spambase_obj = spambase.Spambase()
    spambase_data = spambase_obj.setup_data_spambase(filename=filename, target_class=target_class)

    # Split the data set into 2/3 and 1/3
    spambase_data_train = spambase_data.sample(frac=.667)
    spambase_data_test = spambase_data.drop(spambase_data_train.index)

    # Setup SFS object
    stepwise_forward = sfs.SFS()

    feature_set_columns = list()
    for feature in spambase_data.columns:
        feature_set_columns.append(feature)
    feature_set_columns.remove(target_class)

    spambase_data_train_target_class = spambase_data_train[target_class]
    spambase_data_train_features = spambase_data_train.iloc[:, 0:58]
    spambase_data_test_features = spambase_data_test.iloc[:, 0:58]

    best_features = stepwise_forward.perform_sfs(feature_set_columns=feature_set_columns,
                                                 d_train=spambase_data_train_features,
                                                 d_valid=spambase_data_test_features,
                                                 predicted=spambase_data_train_target_class)

    # Return dataset with best features only
    best_data = spambase_data[best_features]

    return best_features, best_data


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment2Output.txt", "w")

    iris_target_class = "class"
    sfs_iris, sfs_best_data_iris = run_sfs_iris(filename="data/iris.data", target_class=iris_target_class)
    print("Best Feature Selection of Iris: ")
    print(sfs_iris)
    print(sfs_best_data_iris)
    print()
    # Run k means with k=2 since my data setup normalizes data to be either 0 or 1 for the class we intend to have
    kmeans_iris = km.KMeans(num_classes=2, data=sfs_best_data_iris)
    iris_centroids, iris_clusters = kmeans_iris.kmeans_alg()

    iris_coefficient_0, iris_coefficient_1 = kmeans_iris.silhouette_coefficient(iris_clusters)

    # glass_target_class = "Type of glass"
    # sfs_glass, sfs_best_data_glass = run_sfs_glass(filename="data/glass.data", target_class=glass_target_class)
    # print("Best Feature Selection of Glass: ")
    # print(sfs_glass)
    # print(sfs_best_data_glass)
    # print()
    # # Run k means with k=2 since my data setup normalizes data to be either 0 or 1 for the class we intend to have
    # kmeans_glass = km.KMeans(num_classes=2, data=sfs_best_data_glass)
    # glass_centroids, glass_clusters = kmeans_glass.kmeans_alg()
    #
    # spambase_target_class = "57"
    # sfs_spambase, sfs_best_data_spambase = run_sfs_spambase(filename="data/spambase.data", target_class=spambase_target_class)
    # print("Best Feature Selection of Spambase: ")
    # print(sfs_spambase)
    # print(sfs_best_data_spambase)
    # print()
    # # Run k means with k=2 since my data setup normalizes data to be either 0 or 1 for the class we intend to have
    # kmeans_spambase = km.KMeans(num_classes=2, data=sfs_best_data_spambase)
    # spambase_centroids, spambase_clusters = kmeans_spambase.kmeans_alg()


if __name__ == "__main__":
    main()
