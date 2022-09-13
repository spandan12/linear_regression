import math
import numpy as np
from leastsquare import calculate_weight, calculate_error


def k_fold_cross_validate(basis_function, output_labels, K):
    N = len(output_labels)
    test_n = np.floor(N / K)
    train_n = N - test_n

    test_error = np.zeros(K)
    train_error = np.zeros(K)

    for i in range(0, K):
        lower_bound = int(i * test_n)
        upper_bound = int(lower_bound + test_n)
        test_input_data = basis_function[:, lower_bound:upper_bound]
        train_input_data = np.concatenate(
            (basis_function[:, 0:lower_bound], basis_function[:, upper_bound:]), axis=1
        )

        test_output_data = output_labels[lower_bound:upper_bound]
        train_output_data = np.concatenate(
            (output_labels[0:lower_bound], output_labels[upper_bound:]), axis=0
        )

        weight = calculate_weight(train_input_data, train_output_data)
        train_error[i] = calculate_error(train_input_data, train_output_data, weight)
        test_error[i] = calculate_error(test_input_data, test_output_data, weight)

    average_train_error = np.mean(train_error) / train_n
    average_test_error = np.mean(test_error) / test_n

    return average_train_error, average_test_error


if __name__ == "__main__":
    basis_function = np.array(
        [
            [0.23, 0.88, 0.21, 0.92, 0.49, 0.62, 0.77, 0.52, 0.30, 0.19],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    output_labels = np.array(
        [[0.19], [0.96], [0.33], [0.80], [0.46], [0.45], [0.67], [0.32], [0.38], [0.37]]
    )
    k_fold_cross_validate(basis_function, output_labels, 5)
