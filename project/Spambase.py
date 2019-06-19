import pandas as pd
from sklearn import preprocessing


class Spambase:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    class_we_want = "0"

    def setup_data_spambase(self, filename, target_class):
        spambase_names = []
        # Fill soybean_names with column indexes
        for number in range(0, 58):
            spambase_names.append(str(number))

        # Read in data file and turn into data structure
        spambase = pd.read_csv(filename,
                              sep=",",
                              header=0,
                              names=spambase_names)

        # Get copy of data with columns that will be normalized
        new_spambase = spambase[spambase.columns[0:57]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        spambase_scaled_data = scaler.fit_transform(new_spambase)
        # Remove "class" column for now since that column will not be normalized
        spambase_names.remove(target_class)
        spambase_scaled_data = pd.DataFrame(spambase_scaled_data, columns=spambase_names)
        # Add "class" column back to our column list
        spambase_names.append(target_class)

        # Add "class" column into normalized data structure, then categorize it into integers
        spambase_scaled_data[target_class] = spambase[[target_class]]

        # Get mean of each column that will help determine what binary value to turn each into
        spambase_means = spambase_scaled_data.mean()

        # Make categorical column a binary for the class we want to use
        for index, row in spambase_scaled_data.iterrows():
            for column in spambase_names:
                # If the data value is greater than the mean of the column, make it a 1
                if spambase_scaled_data[column][index] > spambase_means[column]:
                    spambase_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    spambase_scaled_data.at[index, column] = 0

        # Return one hot encoded data frame
        return spambase_scaled_data
