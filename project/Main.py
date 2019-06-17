from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Spambase as spambase
from project import Algorithms as alg
import sys


# Naive Bayes for Iris
def run_naive_bayes_iris(filename):
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        iris_obj = iris.Iris()
        iris_data = iris_obj.setup_data_iris(filename=filename)
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        iris_data_train = iris_data.sample(frac=.667)
        iris_data_test = iris_data.drop(iris_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(iris_data_train.iloc[:, 0:4], iris_data_train["class"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(iris_data_test.iloc[:, 0:4], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, iris_data_test["class"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


# Naive Bayes for Glass
def run_naive_bayes_glass(filename):
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        glass_obj = glass.Glass()
        glass_data = glass_obj.setup_data_glass(filename=filename)
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        glass_data_train = glass_data.sample(frac=.667)
        glass_data_test = glass_data.drop(glass_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(glass_data_train.iloc[:, 0:9], glass_data_train["Type of glass"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(glass_data_test.iloc[:, 0:9], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, glass_data_test["Type of glass"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


# Naive Bayes for Spambase
def run_naive_bayes_spambase(filename):
    success_rate = 0
    # Provide 25 iterations
    for index in range(0, 25):
        # Setup data
        spambase_obj = spambase.Spambase()
        spambase_data = spambase_obj.setup_data_spambase(filename=filename)
        # Start algorithm
        naive = alg.Algorithms()
        # Split the data set into 2/3 and 1/3
        spambase_data_train = spambase_data.sample(frac=.667)
        print(spambase_data_train)
        spambase_data_test = spambase_data.drop(spambase_data_train.index)
        # Train classifier
        trainer_classifier = naive.naive_bayes_train(spambase_data_train.iloc[:, 0:58], spambase_data_train["57"])
        # Test classifier
        test_classifier = naive.naive_bayes_test(spambase_data_test.iloc[:, 0:58], trainer_classifier)
        # Get success rate back
        success_rate = success_rate + naive.compare_prediction(test_classifier, spambase_data_test["57"])

    success_rate = success_rate / 25
    print("Average Naive Bayes success rate is: " + str(success_rate) + "%")
    return success_rate


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment2Output.txt", "w")

    # Naive Bayes
    naive_iris = run_naive_bayes_iris(filename="data/iris.data")
    naive_glass = run_naive_bayes_glass(filename="data/glass.data")
    naive_spambase = run_naive_bayes_spambase(filename="data/spambase.data")

    average_naive_combined = (naive_iris + naive_glass + naive_spambase) / 3

    # Statistics from algorithm testing
    print("Overall statistics")
    print()
    print("Naive Bayes iris: " + str(naive_iris) + "%")
    print("Naive Bayes glass: " + str(naive_glass) + "%")
    print("Naive Bayes spambase: " + str(naive_spambase) + "%")
    print()
    print("Average Naive Bayes Accuracy: " + str(average_naive_combined) + "%")


if __name__ == "__main__":
    main()
