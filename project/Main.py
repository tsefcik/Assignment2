from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Spambase as spambase
from project import SFS as sfs
from project import KMeans as km
from project import NaiveBayes as nb
import sys

"""
This is the main driver class for Assignment#2.  @author: Tyler Sefcik
"""


# SFS for Iris
def run_sfs_iris(filename, target_class):
    # Setup data
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris(filename=filename, target_class=target_class)
    # Split the data set into 2/3 and 1/3
    iris_data_train = iris_data.sample(frac=.667)
    iris_data_test = iris_data.drop(iris_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(iris_data_train.iloc[:, 0:4], iris_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(iris_data_test.iloc[:, 0:4], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=iris_data_test[target_class])

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

    return best_features, best_data, iris_data, nb_perf


# SFS for Glass
def run_sfs_glass(filename, target_class):
    # Setup data
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass(filename=filename, target_class=target_class)

    # Split the data set into 2/3 and 1/3
    glass_data_train = glass_data.sample(frac=.667)
    glass_data_test = glass_data.drop(glass_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(glass_data_train.iloc[:, 0:9], glass_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(glass_data_test.iloc[:, 0:9], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=glass_data_test[target_class])

    # Setup SFS object
    stepwise_forward = sfs.SFS()

    # Setup SFS features to go through
    feature_set_columns = list()
    for feature in glass_data.columns:
        feature_set_columns.append(feature)
    feature_set_columns.remove(target_class)

    # Separate data for taining/testing purposes
    glass_data_train_target_class = glass_data_train[target_class]
    glass_data_train_features = glass_data_train.iloc[:, 0:9]
    glass_data_test_features = glass_data_test.iloc[:, 0:9]

    # Run SFS
    best_features = stepwise_forward.perform_sfs(feature_set_columns=feature_set_columns,
                                                 d_train=glass_data_train_features,
                                                 d_valid=glass_data_test_features,
                                                 predicted=glass_data_train_target_class)

    # Return dataset with best features only
    best_data = glass_data[best_features]

    return best_features, best_data, glass_data, nb_perf


# SFS for Spambase
def run_sfs_spambase(filename, target_class):
    # Setup data
    spambase_obj = spambase.Spambase()
    spambase_data = spambase_obj.setup_data_spambase(filename=filename, target_class=target_class)

    # Split the data set into 2/3 and 1/3
    spambase_data_train = spambase_data.sample(frac=.667)
    spambase_data_test = spambase_data.drop(spambase_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(spambase_data_train.iloc[:, 0:58], spambase_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(spambase_data_test.iloc[:, 0:58], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=spambase_data_test[target_class])

    # Setup SFS object
    stepwise_forward = sfs.SFS()

    # Setup SFS features to go through
    feature_set_columns = list()
    for feature in spambase_data.columns:
        feature_set_columns.append(feature)
    feature_set_columns.remove(target_class)

    # Separate data for taining/testing purposes
    spambase_data_train_target_class = spambase_data_train[target_class]
    spambase_data_train_features = spambase_data_train.iloc[:, 0:58]
    spambase_data_test_features = spambase_data_test.iloc[:, 0:58]

    # Run SFS
    best_features = stepwise_forward.perform_sfs(feature_set_columns=feature_set_columns,
                                                 d_train=spambase_data_train_features,
                                                 d_valid=spambase_data_test_features,
                                                 predicted=spambase_data_train_target_class)

    # Return dataset with best features only
    best_data = spambase_data[best_features]

    return best_features, best_data, spambase_data, nb_perf


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    sys.stdout = open("./Assignment2Output.txt", "w")

    ##### Iris #####
    iris_target_class = "class"
    sfs_iris, sfs_best_data_iris, iris_data, iris_nb = run_sfs_iris(filename="data/iris.data", target_class=iris_target_class)
    print("Best Feature Selection of Iris: ")
    print(sfs_iris)
    print(sfs_best_data_iris)
    print()
    # Run k means with k=2 since my data setup normalizes data to be either 0 or 1 for the class we intend to have
    kmeans_iris = km.KMeans(num_classes=2, data=sfs_best_data_iris)
    iris_centroids, iris_clusters = kmeans_iris.kmeans_alg()

    print("Centroids for Iris: ")
    print(iris_centroids)
    print()
    print("Clusters for Iris: ")
    print(iris_clusters)
    print()

    # Get silhouette coefficient
    silhouette_coefficient_iris = kmeans_iris.silhouette_coefficient(iris_clusters)
    print("Silhouette coefficient for Iris: " + str(silhouette_coefficient_iris))
    print()

    # Get success rate for comparison
    compare_iris = nb.NaiveBayes()
    success_rate_iris = compare_iris.compare_prediction(iris_clusters, iris_data[iris_target_class])
    print("Success rate for Iris: " + str(success_rate_iris) + "%")
    print()
    print("Success rate for Iris Naive Bayes: " + str(iris_nb) + "%")
    print('\n' * 3)

    ##### Glass #####
    glass_target_class = "Type of glass"
    sfs_glass, sfs_best_data_glass, glass_data, glass_nb = run_sfs_glass(filename="data/glass.data", target_class=glass_target_class)
    print("Best Feature Selection of Glass: ")
    print(sfs_glass)
    print(sfs_best_data_glass)
    print()
    # Run k means with k=2 since my data setup normalizes data to be either 0 or 1 for the class we intend to have
    kmeans_glass = km.KMeans(num_classes=2, data=sfs_best_data_glass)
    glass_centroids, glass_clusters = kmeans_glass.kmeans_alg()

    print("Centroids for Glass: ")
    print(glass_centroids)
    print()
    print("Clusters for Glass: ")
    print(glass_clusters)
    print()

    # Get silhouette coefficient
    silhouette_coefficient_glass = kmeans_glass.silhouette_coefficient(glass_clusters)
    print("Silhouette coefficient for Glass: " + str(silhouette_coefficient_glass))
    print()

    # Get success rate for comparison
    compare_glass = nb.NaiveBayes()
    success_rate_glass = compare_glass.compare_prediction(glass_clusters, glass_data[glass_target_class])
    print("Success rate for Glass: " + str(success_rate_glass) + "%")
    print()
    print("Success rate for Glass Naive Bayes: " + str(glass_nb) + "%")
    print('\n' * 3)

    ##### Spambase #####
    spambase_target_class = "57"
    sfs_spambase, sfs_best_data_spambase, spambase_data, spambase_nb = run_sfs_spambase(filename="data/spambase.data",
                                                                                        target_class=spambase_target_class)
    print("Best Feature Selection of Spambase: ")
    print(sfs_spambase)
    print(sfs_best_data_spambase)
    print()
    # Run k means with k=2 since my data setup normalizes data to be either 0 or 1 for the class we intend to have
    kmeans_spambase = km.KMeans(num_classes=2, data=sfs_best_data_spambase)
    spambase_centroids, spambase_clusters = kmeans_spambase.kmeans_alg()

    print("Centroids for Spambase: ")
    print(spambase_centroids)
    print()
    print("Clusters for Spambase: ")
    print(spambase_clusters)
    print()

    # Get silhouette coefficient
    silhouette_coefficient_spambase = kmeans_spambase.silhouette_coefficient(spambase_clusters)
    print("Silhouette coefficient for Spambase: " + str(silhouette_coefficient_spambase))
    print()

    # Get success rate for comparison
    compare_spambase = nb.NaiveBayes()
    success_rate_spambase = compare_spambase.compare_prediction(spambase_clusters, spambase_data[spambase_target_class])
    print("Success rate for Spambase: " + str(success_rate_spambase) + "%")
    print()
    print("Success rate for Spambase Naive Bayes: " + str(spambase_nb) + "%")
    print('\n' * 3)


if __name__ == "__main__":
    main()
