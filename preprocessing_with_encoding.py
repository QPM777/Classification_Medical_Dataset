import os as os
import numpy as np
from preprocessing import *

# Specific pre-processing function for the third model, that use categorical feature encoding.


def standardize_num(data, mean, std, categorical_flags):
    for col in range(len(categorical_flags)):
        if not categorical_flags[col]:
            data[:, col] = (data[:, col] - mean[col]) / std[col]
    return data


# Compute the mode of a 1D NumPy array
def mode(arr):
    # Exclude NaN values
    vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
    # Return the mode
    return vals[np.argmax(counts)]


def prepare_train_data(x, degree=1):

    new_x = x.copy()

    new_x, rm_nan_columns_idx = remove_nan_columns(new_x, nan_percentage=80)

    categorical_flags = categorize_columns(new_x)
    trained_imputer = train_imputer(new_x, categorical_flags)
    new_x = apply_imputer(new_x, categorical_flags, trained_imputer)
    new_x = shift_and_map_categorical_columns(new_x, categorical_flags)

    encoder = CountFrequencyEncoder(flags=categorical_flags)
    encoder.fit(new_x)
    new_x = encoder.transform(new_x)

    new_x = build_poly(new_x, degree=degree)

    scaling_mean = np.nanmean(new_x, axis=0)
    scaling_std = np.where(np.nanstd(new_x, axis=0) == 0, 1, np.nanstd(new_x, axis=0))

    updated_categorical_flags = categorize_columns(new_x)

    new_w = standardize_num(new_x, scaling_mean, scaling_std, updated_categorical_flags)
    new_x = np.hstack((np.ones((new_x.shape[0], 1)), new_x))

    return (
        new_x,
        scaling_mean,
        scaling_std,
        degree,
        rm_nan_columns_idx,
        categorical_flags,
        updated_categorical_flags,
        trained_imputer,
        encoder,
    )


def prepare_test_data(
    x,
    scaling_mean,
    scaling_std,
    degree,
    rm_col_index,
    categorical_flags,
    updated_categorical_flags,
    trained_imputer,
    encoder,
):

    new_x = x.copy()
    new_x = np.delete(new_x, rm_col_index, 1)
    new_x = apply_imputer(new_x, categorical_flags, trained_imputer)
    new_x = shift_and_map_categorical_columns(new_x, categorical_flags)

    new_x = encoder.transform(new_x)

    new_x = build_poly(new_x, degree=degree)
    new_w = standardize_num(new_x, scaling_mean, scaling_std, updated_categorical_flags)
    new_x = np.hstack((np.ones((x.shape[0], 1)), new_x))

    return new_x


# Shift and map values of categorical columns in x_train.
def shift_and_map_categorical_columns(x_train, categorical_flags):

    # Copy the original data to avoid modifying it directly
    updated_x_train = np.copy(x_train)

    # Loop through each column
    for idx in range(updated_x_train.shape[1]):
        if categorical_flags[idx]:  # Check if the column is categorical
            # Get unique values
            unique_values = np.unique(updated_x_train[:, idx])

            # Create a mapping from old values to new values (0, 1, 2, ...)
            mapping = {
                old_val: new_val for new_val, old_val in enumerate(unique_values)
            }

            # Transform the column using the mapping
            updated_x_train[:, idx] = np.vectorize(mapping.get)(updated_x_train[:, idx])

    return updated_x_train


# Determines if a numerical column should be treated as categorical based on its unique values.
def is_numerical_categorical(column, unique_threshold):

    # Remove any missing or empty values (if represented as strings)
    cleaned_column = column[~np.isnan(column)]

    total_values = len(cleaned_column)

    # Count unique values
    unique_values = np.unique(cleaned_column)
    unique_count = len(unique_values)

    # Calculate the ratio of unique values to total values
    unique_ratio = unique_count / total_values

    # If the ratio of unique values is small, consider it categorical
    # return unique_ratio < unique_threshold

    return unique_count < 55


# Determines if columns in a dataset are categorical or numeric.
def categorize_columns(
    data, unique_threshold=0.0002
):  # 55 states (different values) corresponds nearly to 0.02% of the rows

    categorical_flags = []
    for col in range(data.shape[1]):
        categorical_flags.append(
            is_numerical_categorical(data[:, col], unique_threshold)
        )

    return categorical_flags


def train_imputer(data, categorical_flags):

    trained_imputer = []
    # Iterate through each column
    for col in range(data.shape[1]):
        if categorical_flags[col]:
            # Handle categorical columns: impute with mode
            mode_value = mode(data[:, col])
            trained_imputer.append(mode_value)
        else:
            # Handle numeric columns: impute with median
            mean_value = np.nanmedian(data[:, col])
            trained_imputer.append(mean_value)

    return trained_imputer


def apply_imputer(data, categorical_flags, trained_imputer):

    # Make a copy of the data to avoid modifying the original
    imputed_data = data.copy()
    # Iterate through each column
    for col in range(data.shape[1]):
        if categorical_flags[col]:
            # Handle categorical columns: impute with mode
            mode_value = trained_imputer[col]
            imputed_data[:, col] = np.where(
                np.isnan(imputed_data[:, col]), mode_value, imputed_data[:, col]
            )
        else:
            # Handle numeric columns: impute with median
            mean_value = trained_imputer[col]
            # Impute missing values with median
            imputed_data[:, col] = np.where(
                np.isnan(imputed_data[:, col]), mean_value, imputed_data[:, col]
            )

    return imputed_data


class CountFrequencyEncoder:
    def __init__(self, encoding_method="frequency", flags=None):
        self.encoding_method = encoding_method
        self.flags = flags  # Boolean array indicating categorical columns
        self.encoder_dict_ = {}

    def fit(self, X):
        """Fit the encoder to the data, learning the frequencies."""
        for col in range(X.shape[1]):
            if self.flags[col]:  # Only process if the column is categorical
                unique, counts = np.unique(X[:, col], return_counts=True)
                frequencies = counts / counts.sum()  # Calculate frequency

                # Store frequencies in a dictionary
                self.encoder_dict_[col] = {
                    str(u): f for u, f in zip(unique, frequencies)
                }

                # Add 'Other' for levels not in the top 5
                sorted_items = sorted(
                    self.encoder_dict_[col].items(), key=lambda x: x[1], reverse=True
                )
                top_levels = {k: v for k, v in sorted_items[:5]}
                other_count = 1 - sum(
                    top_levels.values()
                )  # Remaining frequency for 'Other'
                top_levels["Other"] = other_count

                self.encoder_dict_[col] = top_levels  # Update to include 'Other'

    def transform(self, X):
        """Transform the data using the learned frequencies."""
        X_transformed = X.copy()
        for col in range(X.shape[1]):
            if self.flags[col]:  # Only transform categorical columns
                # Map the frequencies based on encoder_dict_
                X_transformed[:, col] = np.vectorize(self._map_frequency)(
                    X_transformed[:, col], self.encoder_dict_[col]
                )
        return X_transformed

    def _map_frequency(self, value, frequency_dict):
        """Helper function to map a value to its frequency."""
        return frequency_dict.get(str(value), 0)  # Default to 0 for unseen categories
