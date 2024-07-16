import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split


def clean_dataset(data_file, y_col):
    # Loads the first 10,000 data
    dataset = pd.read_csv(data_file).dropna().head(10000)

    for column in dataset:
        temp = list(set(dataset[column]))
        temp.sort()
        if isinstance(temp[0], str):

            dictionary = {}
            for index, value in enumerate(temp):
                dictionary[value] = index

            dataset[column] = dataset[column].map(dictionary)

    return split_dataset(dataset, y_col)


def split_dataset(dataset, y_col):
    y = dataset[y_col]
    X = dataset.drop(y_col, axis=1)

    return train_test_split(X, y, test_size=0.2, shuffle=True)
