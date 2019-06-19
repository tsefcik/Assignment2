import pandas as pd
import numpy as np


class NaiveBayes:

    # Method used to train the naive bayes classifier
    def naive_bayes_train(self, data, predicted):
        # Get column list for output matrix
        col_list = list(data.columns)
        col_list.insert(0, "Overall Mean")
        output = pd.DataFrame(columns=[col_list])
        overall_mean = np.mean(predicted)

        # Set initial values
        output.loc[0] = (1 - overall_mean)
        output.loc[1] = (1 - overall_mean)
        output.loc[2] = overall_mean
        output.loc[3] = overall_mean

        # Split classes into true/false
        class_false = data[predicted == 0]
        class_true = data[predicted == 1]

        # Add a middle point that will give us a balance point
        new_row = [.5] * (class_false.shape[1])
        class_false = class_false.append(pd.Series(new_row, index=class_false.columns), ignore_index=True)
        class_true = class_true.append(pd.Series(new_row, index=class_true.columns), ignore_index=True)

        # Get the mean of each class
        class_false_means = class_false.mean()
        class_true_means = class_true.mean()

        adjusted_false = 1 - class_false_means
        adjusted_true = 1 - class_true_means

        # Set the output matrix with the means of the columns
        for col in col_list:
            if col != "Overall Mean":
                for index in range(0, len(class_false_means)):
                    output.loc[0][index+1] = adjusted_false[index]
                    output.loc[1][index+1] = class_false_means[index]
                    output.loc[2][index+1] = adjusted_true[index]
                    output.loc[3][index+1] = class_true_means[index]

        return output

    # Method used to test our naive bayes classifier
    def naive_bayes_test(self, data, mean_matrix):
        predictions = []
        # Iterate through each row, and set initial output classifier
        for index, row in data.iterrows():
            class_false = [0] * (len(data.columns) + 1)
            class_false[0] = mean_matrix.iloc[0, 0]
            class_true = [0] * (len(data.columns) + 1)
            class_true[0] = mean_matrix.iloc[0, 0]

            # Iterate through each column in the row
            for index2 in range(0, len(data.columns)):
                # If the column is equal to 1, set the mean accordingly
                if row[index2] == 1:
                    class_false[index2 + 1] = mean_matrix.iloc[1, index2 + 1]
                    class_true[index2 + 1] = mean_matrix.iloc[3, index2 + 1]
                # If the column is equal to 0, set the mean accordingly
                else:
                    class_false[index2 + 1] = mean_matrix.iloc[0, index2 + 1]
                    class_true[index2 + 1] = mean_matrix.iloc[2, index2 + 1]

            # Take the product of each dataframe
            class_false_product = np.product(class_false)
            class_true_product = np.product(class_true)

            # Compare to give us our prediction
            if class_false_product < class_true_product:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    # Used to compare if a class was predicted correctly
    def compare_prediction(self, predict_classier, data):
        success = 0
        # Iterate through predictions and see if there is a match
        for index in range(0, len(predict_classier) - 1):

            if predict_classier[index] == data.values[index]:
                success = success + 1

        success_rate = (success / data.shape[0]) * 100
        # Return the success rate
        return success_rate
