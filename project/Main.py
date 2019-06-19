from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Spambase as spambase
from project import SFS as sfs
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

    feature_set_columns = list()
    for feature in iris_data.columns:
        feature_set_columns.append(feature)
    feature_set_columns.remove(target_class)

    iris_data_train_target_class = iris_data_train[target_class]
    iris_data_train_features = iris_data_train.iloc[:, 0:4]
    iris_data_test_features = iris_data_test.iloc[:, 0:4]

    best_features = stepwise_forward.perform_sfs(feature_set_columns=feature_set_columns,
                                                 d_train=iris_data_train_features,
                                                 d_valid=iris_data_test_features,
                                                 predicted=iris_data_train_target_class)
    return best_features


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
    return best_features


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
    return best_features


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment2Output.txt", "w")

    iris_target_class = "class"
    sfs_iris = run_sfs_iris(filename="data/iris.data", target_class=iris_target_class)
    print("Best Feature Selection of Iris: ")
    print(sfs_iris)
    print()

    glass_target_class = "Type of glass"
    sfs_glass = run_sfs_glass(filename="data/glass.data", target_class=glass_target_class)
    print("Best Feature Selection of Glass: ")
    print(sfs_glass)
    print()

    spambase_target_class = "57"
    sfs_spambase = run_sfs_spambase(filename="data/spambase.data", target_class=spambase_target_class)
    print("Best Feature Selection of Spambase: ")
    print(sfs_spambase)
    print()

    # # Naive Bayes
    # naive_iris = run_naive_bayes_iris(filename="data/iris.data")
    # naive_glass = run_naive_bayes_glass(filename="data/glass.data")
    # naive_spambase = run_naive_bayes_spambase(filename="data/spambase.data")


if __name__ == "__main__":
    main()
