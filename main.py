import numpy as np

from readfile import read_train_data
from crossvalidate import k_fold_cross_validate


if __name__ == "__main__":
    train_input_data, train_label_data = read_train_data()
    basis_function = np.transpose(train_input_data)
    output_labels = train_label_data.reshape(-1, 1)
    K = 60

    average_train_error, average_test_error = k_fold_cross_validate(
        basis_function, output_labels, K
    )

    print(
        "average_train_error:",
        average_train_error,
        "\naverage_test_error:",
        average_test_error,
    )
