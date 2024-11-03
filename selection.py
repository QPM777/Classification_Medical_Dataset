import numpy as np
from train import *
from evaluation import *

from mpl_toolkits.mplot3d import Axes3D
from train import predict_with_voting

import preprocessing_with_encoding


def plot_f1_accuracy_3d(
    X_train,
    X_test,
    y_train,
    y_test,
    thresholds,
    max_degree=6,
    synt_data_method=False,
    cat_encoding=False,
):
    # Lists to store data for plotting
    degrees = []
    threshold_values = []
    accuracy_values = []
    f1_values = []

    # Loop through polynomial degrees and thresholds
    for degree in range(2, max_degree + 1):  # Polynomial degrees from 2 to 6
        print(f"  Degree = {degree}")
        if cat_encoding:
            (
                X_train_prepared,
                scaling_mean,
                scaling_std,
                degree,
                rm_nan_columns_idx,
                categorical_flags,
                updated_categorical_flags,
                trained_imputer,
                encoder,
            ) = preprocessing_with_encoding.prepare_train_data(X_train, degree=degree)
            X_test_prepared = preprocessing_with_encoding.prepare_test_data(
                X_test,
                scaling_mean,
                scaling_std,
                degree,
                rm_nan_columns_idx,
                categorical_flags,
                updated_categorical_flags,
                trained_imputer,
                encoder,
            )
        else:
            (
                X_train_prepared,
                mean,
                scaling_mean,
                scaling_std,
                degree,
                rm_nan_columns_idx,
            ) = prepare_train_data(X_train, degree=degree)
            X_test_prepared = prepare_test_data(
                X_test, mean, scaling_mean, scaling_std, degree, rm_nan_columns_idx
            )

        if synt_data_method:
            models = train_models_with_synt_data(X_train_prepared, y_train)
        else:
            models = train_models(X_train_prepared, y_train)

        for threshold in thresholds:  # Iterate over specified thresholds

            y_pred_test = predict_with_voting(models, X_test_prepared, threshold)

            # Compute metrics
            accuracy = compute_accuracy(y_test, y_pred_test)
            f1_score = compute_f1_score(y_test, y_pred_test)

            # Store results
            degrees.append(degree)
            threshold_values.append(threshold)
            accuracy_values.append(accuracy)
            f1_values.append(f1_score)

    # Convert lists to numpy arrays for plotting
    degrees = np.array(degrees)
    threshold_values = np.array(threshold_values)
    accuracy_values = np.array(accuracy_values)
    f1_values = np.array(f1_values)

    # Find the optimal point with the maximum accuracy and F1-score
    max_accuracy_idx = np.argmax(accuracy_values)
    max_f1_idx = np.argmax(f1_values)

    # Plotting
    fig = plt.figure(figsize=(14, 8))

    # Plot for Accuracy
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(degrees, threshold_values, accuracy_values, c="b", label="Accuracy")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Threshold")
    ax.set_zlabel("Accuracy")
    ax.set_title("3D Plot of Accuracy vs Polynomial Degree and Threshold")

    # Highlight the optimal point with the highest accuracy
    ax.scatter(
        degrees[max_accuracy_idx],
        threshold_values[max_accuracy_idx],
        accuracy_values[max_accuracy_idx],
        color="orange",
        s=100,
        label="Highest Accuracy",
        edgecolor="black",
    )
    ax.legend()

    # Plot for F1 Score
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(degrees, threshold_values, f1_values, c="r", label="F1 Score")
    ax2.set_xlabel("Polynomial Degree")
    ax2.set_ylabel("Threshold")
    ax2.set_zlabel("F1 Score")
    ax2.set_title("3D Plot of F1 Score vs Polynomial Degree and Threshold")

    # Highlight the optimal point with the lowest F1-score
    ax2.scatter(
        degrees[max_f1_idx],
        threshold_values[max_f1_idx],
        f1_values[max_f1_idx],
        color="purple",
        s=100,
        label="Highest F1 Score",
        edgecolor="black",
    )
    ax2.legend()

    print("")
    print(
        f"The combination (degree = {degrees[max_f1_idx]},threshold = {threshold_values[max_f1_idx]}) \n is the optimal one and leads to an f1_score of {f1_values[max_f1_idx]} and an accuracy of {accuracy_values[max_f1_idx]}"
    )
    print("")

    plt.tight_layout()
    plt.show()


# def GridSearch(X_train, X_test, y_train, y_test):

#     degrees = range(2, 5)
#     best_f1_score = -1
#     best_param = []

#     for degree in degrees:

#         print(f'GridSearch degree = {degree}')

#         X_train_prepared, mean, scaling_mean, scaling_std, degree, rm_nan_columns_idx = prepare_train_data(X_train, degree=degree)
#         X_test_prepared = prepare_test_data(X_test, mean, scaling_mean, scaling_std, degree, rm_nan_columns_idx)
#         models = train_models(X_train_prepared, y_train)
#         best_threshold = find_best_treshold(models, X_test_prepared, y_test)
#         y_pred = predict_with_voting(models, X_test_prepared, threshold=best_threshold)
#         f1_score = compute_f1_score(y_test, y_pred)

#         if f1_score > best_f1_score:
#             best_f1_score = f1_score
#             best_param = [degree, best_threshold]

#     print(f'best_param : degree = {best_param[0]}, threshold = {best_param[1]}')

#     return best_param

# def find_best_treshold(model, X_test, y_test):
#     thresholds = np.linspace(0, 1, 100)
#     max_f1_score = -1
#     best_threshold = -1
#     for threshold in thresholds:
#         y_pred = predict_with_voting(model, X_test, threshold)
#         if compute_f1_score(y_test, y_pred) > max_f1_score:
#             max_f1_score = compute_f1_score(y_test, y_pred)
#             best_threshold = threshold
#     return best_threshold

# def find_best_degree(X_train, X_test, y_train, y_test):
#     degrees = range(2, 6)
#     f1_scores = []

#     for degree in degrees:

#         X_train_prepared, mean, scaling_mean, scaling_std, degree, rm_nan_columns_idx = prepare_train_data(X_train, degree=degree)
#         X_test_prepared = prepare_test_data(X_test, mean, scaling_mean, scaling_std, degree, rm_nan_columns_idx)

#         models = train_models(X_train_prepared, y_train)
#         y_pred = predict_with_voting(models, X_test_prepared, threshold=0.694)

#         f1_scores.append(compute_f1_score(y_test, y_pred))

#     plt.figure(figsize=(8, 6))
#     plt.bar(degrees, f1_scores, color='skyblue')
#     plt.xlabel('Degree')
#     plt.ylabel('F1 Score')
#     plt.title('F1 Score for Different Polynomial Degrees')
#     plt.xticks(degrees)
#     plt.ylim(0, 0.6)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()

#     # Retourne le meilleur degré pour référence si besoin
#     best_degree = degrees[np.argmax(f1_scores)]
#     print(f'Best degree: {best_degree} with F1 score: {max(f1_scores):.4f}')
#     return best_degree
