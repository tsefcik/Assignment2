import pandas as pd
from sklearn import preprocessing


class Iris:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    class_we_want = "Iris-virginica"

    def setup_data_iris(self, filename):
        # Read in data file and turn into data structure
        iris = pd.read_csv(filename,
                           sep=",",
                           header=0,
                           names=self.iris_names)

        # Make categorical column a binary for the class we want to use
        for index, row in iris.iterrows():
            if iris["class"][index] == self.class_we_want:
                iris.at[index, "class"] = 1
            else:
                iris.at[index, "class"] = 0
        # Get copy of data with columns that will be normalized
        new_iris = iris[iris.columns[0:4]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        iris_scaled_data = scaler.fit_transform(new_iris)
        # Remove "class" column for now since that column will not be normalized
        self.iris_names.remove("class")
        iris_scaled_data = pd.DataFrame(iris_scaled_data, columns=self.iris_names)
        # Add "class" column back to our column list
        self.iris_names.append("class")

        # Add "class" column into normalized data structure, then categorize it into integers
        iris_scaled_data["class"] = iris[["class"]]

        # Get mean of each column that will help determine what binary value to turn each into
        iris_means = iris_scaled_data.mean()

        # Make categorical column a binary for the class we want to use
        for index, row in iris_scaled_data.iterrows():
            for column in self.iris_names:
                # If the data value is greater than the mean of the column, make it a 1
                if iris_scaled_data[column][index] > iris_means[column]:
                    iris_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    iris_scaled_data.at[index, column] = 0

        # Return one hot encoded data frame
        return iris_scaled_data
