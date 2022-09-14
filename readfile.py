import pandas as pd


def read_file(file_path):

    return pd.read_csv(file_path, sep="\s+", header=None)


def get_input_and_labels(train_df):
    return train_df.iloc[:, :-1].to_numpy(), train_df.iloc[:, -1].to_numpy()


def read_train_data():
    df = read_file("./data/traindata.txt")

    return get_input_and_labels(df)


def read_test_data():

    return read_file("./data/testinputs.txt").to_numpy()


if __name__ == "__main__":
    read_train_data()
    read_test_data()
