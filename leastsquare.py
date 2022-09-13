import numpy as np


def calculate_weight(basis_function, output_labels):

    # For MW = S
    S = basis_function @ output_labels
    M = basis_function @ (basis_function.T)

    # Solve linear equation
    return np.linalg.solve(M, S)


def calc_predictions(basis_function, weight):
    return (weight.T) @ basis_function


def calculate_error(predicted_output, output_labels):
    return np.linalg.norm(predicted_output - output_labels.T) ** 2


if __name__ == "__main__":
    basis_function = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.7]])
    output_labels = np.array([[0.9], [1.0], [1.1], [1.2]])
    weight = calculate_weight(basis_function, output_labels)
    predicted_output = calc_predictions(basis_function, weight)
    print(calculate_error(predicted_output, output_labels))
