from project import NaiveBayes as nb
import numpy as np


class SFS:

    """
    Creates a NaiveBayes object to run Naive Bayes algorithms later.
    """
    def __init__(self):
        self.nb = nb.NaiveBayes()

    """
    Algorithm to perform Stepwise Forward Selection with Naive Bayes classifier.
    feature_set_columns: set of features we will be iterating through from dataset
    d_train: set of training data from dataset
    d_valid: set of test data from dataset
    predicted: class that we want to compare our prediction to for accuracy
    """
    def perform_sfs(self, feature_set_columns, d_train, d_valid, predicted):
        # Keep track of best feature set
        best_features = list()
        # Set base performance to a low negative number for initialization
        base_perf = -np.Infinity

        # Keep iterating until we have gone through all features in the dataset
        while len(feature_set_columns) > 0:
            # Set best performance to a low negative number for initialization
            best_perf = -np.Infinity

            # Iterate through features in dataset
            for feature in feature_set_columns:
                # Add the feature to the result list
                best_features.append(feature)

                d_train = d_train[feature_set_columns]
                d_valid = d_valid[feature_set_columns]

                # Get Naive Bayes classifier with given training data and wanted class
                classifier = self.nb.naive_bayes_train(data=d_train, predicted=predicted)
                # Get the predictions from the Naive Bayes test with valid data and classifier from above
                predictions = self.nb.naive_bayes_test(data=d_valid, mean_matrix=classifier)
                # Calculate performance by returning success rate
                curr_perf = self.nb.compare_prediction(predict_classier=predictions, data=predicted)

                # Compare the current performance to the best performance so far
                # Set variables accordingly
                if curr_perf > best_perf:
                    best_perf = curr_perf
                    best_feature = feature

                # Remove current feature from best features list
                best_features.remove(feature)

            # If the best_perf is better than or equal to(I was getting a few ties and needed to break them, so I just
            # included both as valuable features) the base_perf, we add the feature to the best features list
            if best_perf >= base_perf:
                base_perf = best_perf
                feature_set_columns.remove(best_feature)
                best_features.append(best_feature)
            else:
                break

        # Return list of best features
        return best_features
