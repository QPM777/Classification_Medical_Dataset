import os as os
import numpy as np


def build_poly(x, degree=1):
    poly = None
    for deg in range(1, degree + 1):
        poly = np.power(x, deg) if poly is None else np.c_[poly, np.power(x, deg)]
    return poly


def standardize(data, mean, std):
    return (data - mean) / std


def replace_nan_mean(x, mean):
    for col in range(x.shape[1]):
        col_values = x[:, col]
        x[col_values != col_values, col] = mean[col]
    return x


def remove_nan_columns(x, nan_percentage=80.0):
    rows = x.shape[0]
    idx_to_drop = [
        col
        for col in range(x.shape[1])
        if np.sum(x[:, col] != x[:, col]) / rows * 100 > nan_percentage
    ]
    return x[:, [i for i in range(x.shape[1]) if i not in idx_to_drop]], idx_to_drop


def remove_nan_rows(x, nan_percentage=50.0):
    cols = x.shape[1]
    idx_to_drop = [
        row
        for row in range(x.shape[0])
        if np.sum(x[row, :] != x[row, :]) / cols * 100 > nan_percentage
    ]
    return x[[i for i in range(x.shape[0]) if i not in idx_to_drop], :], idx_to_drop


def prepare_train_data(x, degree=1):
    new_x = x.copy()
    new_x, rm_nan_columns_idx = remove_nan_columns(x, nan_percentage=80)
    mean = np.nanmean(new_x, axis=0)
    new_x = replace_nan_mean(new_x, mean)
    new_x = build_poly(new_x, degree=degree)
    scaling_mean = np.nanmean(new_x, axis=0)
    scaling_std = np.where(np.nanstd(new_x, axis=0) == 0, 1, np.nanstd(new_x, axis=0))
    new_x = standardize(new_x, scaling_mean, scaling_std)
    new_x = np.hstack((np.ones((new_x.shape[0], 1)), new_x))
    return new_x, mean, scaling_mean, scaling_std, degree, rm_nan_columns_idx


def prepare_test_data(x, mean, scaling_mean, scaling_std, degree, rm_col_index):
    new_x = x.copy()
    new_x = np.delete(x, rm_col_index, 1)
    new_x = replace_nan_mean(new_x, mean)
    new_x = build_poly(new_x, degree=degree)
    new_x = standardize(new_x, scaling_mean, scaling_std)
    new_x = np.hstack((np.ones((x.shape[0], 1)), new_x))
    return new_x


def create_balanced_datasets_no_overlap(X_train, y_train, n_datasets=9):

    # Separate minority (label 1) and majority (label -1) classes
    minority_class_X = X_train[y_train == 1]
    majority_class_X = X_train[y_train == -1]

    minority_class_y = y_train[y_train == 1]
    majority_class_y = y_train[y_train == -1]

    # Number of samples per split for the majority class (10% from the majority class vs 8.83% of the minority )
    split_size = int(0.1 * X_train.shape[0])

    # Shuffle the majority class to ensure randomness
    shuffled_indices = np.random.permutation(majority_class_X.shape[0])
    majority_class_X = majority_class_X[shuffled_indices]
    majority_class_y = majority_class_y[shuffled_indices]

    balanced_datasets = []

    # Create n_datasets balanced datasets
    for i in range(n_datasets):
        # Determine the start and end of the current split
        start_idx = i * split_size
        end_idx = (i + 1) * split_size

        # Get the current 10% split from the majority class
        current_majority_X = majority_class_X[start_idx:end_idx]
        current_majority_y = majority_class_y[start_idx:end_idx]

        # Combine minority class with the current split of majority class
        balanced_X = np.vstack((minority_class_X, current_majority_X))
        balanced_y = np.hstack((minority_class_y, current_majority_y))

        # Shuffle the combined dataset
        shuffled_indices = np.random.permutation(len(balanced_y))
        balanced_X = balanced_X[shuffled_indices]
        balanced_y = balanced_y[shuffled_indices]
        balanced_y = np.expand_dims(balanced_y, 1)

        # Add the balanced dataset to the list
        balanced_datasets.append((balanced_X, balanced_y))

    return balanced_datasets


def initialize_weights(x):
    return np.random.random((x.shape[1], 1))


def generate_synthetic_points_with_boundary_focus(
    minority_class_points, majority_class_points, n_samples, alpha_range=(0.3, 0.9)
):
    """
    Generate synthetic samples for the minority class using a convex combination
    of each point, the minority class centroid, and the majority class centroid.
    """
    # Compute centroids of the minority and majority classes
    centroid_minority = np.mean(minority_class_points, axis=0)
    centroid_majority = np.mean(majority_class_points, axis=0)

    synthetic_samples = []

    for _ in range(n_samples):
        # Select a random point from the minority class
        idx = np.random.randint(0, minority_class_points.shape[0])
        x_real = minority_class_points[idx]

        # Randomly choose alpha and beta within specified ranges
        alpha = np.random.uniform(*alpha_range)

        # Generate the synthetic point using a weighted combination
        x_synthetic = (
            alpha * x_real + (1 - alpha) * (centroid_minority + centroid_majority) / 2
        )

        synthetic_samples.append(x_synthetic)

    return np.array(synthetic_samples)


def create_balanced_datasets_with_synthetic_data(X_train, y_train, n_datasets=9):
    """
    Create balanced datasets by combining the minority class with synthetic points
    and equal-sized splits of the majority class, without overlap in majority samples.
    """
    # Separate minority (label 1) and majority (label -1) classes
    minority_class_X = X_train[y_train == 1]
    majority_class_X = X_train[y_train == -1]

    minority_class_y = y_train[y_train == 1]
    majority_class_y = y_train[y_train == -1]

    # Calculate the number of synthetic samples needed for balance
    split_size = int((1.5) * len(minority_class_X))  # equal number to minority class

    # Shuffle the majority class to ensure randomness
    shuffled_indices = np.random.permutation(majority_class_X.shape[0])
    majority_class_X = majority_class_X[shuffled_indices]
    majority_class_y = majority_class_y[shuffled_indices]

    balanced_datasets = []

    # Create n_datasets balanced datasets
    for i in range(n_datasets):
        # Determine the start and end of the current split
        start_idx = i * split_size
        end_idx = (
            (i + 1) * split_size
            if (i + 1) * split_size < len(majority_class_X)
            else len(majority_class_X)
        )

        # Get the current split from the majority class
        current_majority_X = majority_class_X[start_idx:end_idx]
        current_majority_y = majority_class_y[start_idx:end_idx]

        # Generate synthetic points for the minority class to match the majority split size
        # n_synthetic_samples = split_size - len(minority_class_X)
        synthetic_minority_X = generate_synthetic_points_with_boundary_focus(
            minority_class_X, majority_class_X, split_size // 3
        )

        # Combine real and synthetic minority points
        combined_minority_X = np.vstack((minority_class_X, synthetic_minority_X))
        combined_minority_y = np.ones(combined_minority_X.shape[0])

        # Combine with the current split of the majority class
        balanced_X = np.vstack((combined_minority_X, current_majority_X))

        balanced_y = np.hstack((combined_minority_y, current_majority_y))

        # Shuffle the combined dataset
        shuffled_indices = np.random.permutation(len(balanced_y))
        balanced_X = balanced_X[shuffled_indices]
        balanced_y = balanced_y[shuffled_indices]
        balanced_y = np.expand_dims(balanced_y, 1)

        # Add the balanced dataset to the list
        balanced_datasets.append((balanced_X, balanced_y))

    return balanced_datasets
