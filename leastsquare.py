import numpy as np


def calculate_weight(basis_function, output_labels):

    # For MW = S
    S = basis_function@output_labels
    M = basis_function@(basis_function.T)

    # Solve linear equation
    return np.linalg.solve(M, S)


def calculate_error(basis_function, output_labels, weight):
    predicted_output = (weight.T)@basis_function
    rms = np.sum((predicted_output - output_labels.T) ** 2)

    return rms


if __name__ == "__main__":
    basis_function = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.7]])
    output_labels = np.array([[0.9], [1.0], [1.1], [1.2]])
    weight = calculate_weight(basis_function, output_labels)
    print(calculate_error(basis_function, output_labels, weight))
