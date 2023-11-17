
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_vbs(cost_matrix):
    # Find the index of the best algorithm for each instance
    best_algorithms = np.argmin(cost_matrix, axis=1)

    # Calculate the average cost of selecting the best algorithm for each instance
    vbs_average_costs = [cost_matrix[i][best_algorithms[i]] for i in range(len(cost_matrix))]

    return np.mean(vbs_average_costs)


def calculate_sbs(cost_matrix):
    # Calculate the column averages
    column_averages = np.mean(cost_matrix, axis=0)

    # Find the column with the lowest average
    lowest_average_column_index = np.argmin(column_averages)

    return column_averages[lowest_average_column_index]


def read_data(dataset_dir):
    """Read the training and the test data of an AS problem

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset folder.

    Returns
    -------
    train_performance_data : {ndarray} of shape (#instances, #algorithms)
        Cost of solving each instance in the training set by each algorithm.
    train_instance_features : {ndarray} of shape (#instances, #features)
        Features of the training instances.
    test_performance_data : {ndarray} of shape (#instances, #algorithms)
        Cost of solving each instance in the test set by each algorithm.
    test_instance_features : {ndarray} of shape (#instances, #features)
        Features of the test instances.
    """     
    train_performance_data = np.genfromtxt(f"{dataset_dir}/train/performance-data.txt", delimiter=",")
    train_instance_features = np.genfromtxt(f"{dataset_dir}/train/instance-features.txt", delimiter=",") 
    test_performance_data = np.genfromtxt(f"{dataset_dir}/test/performance-data.txt", delimiter=",")
    test_instance_features = np.genfromtxt(f"{dataset_dir}/test/instance-features.txt", delimiter=",")   
    return train_performance_data, train_instance_features, test_performance_data, test_instance_features


def get_vbs_sbs(train_performance_data, test_performance_data):
    """Get the VBS and the SBS

    Parameters
    ----------
    train_performance_data : {ndarray} of shape (#instances, #algorithms)
        Cost of solving each instance in the training set by each algorithm.
    test_performance_data : {ndarray} of shape (#instances, #algorithms)
        Cost of solving each instance in the test set by each algorithm.

    Returns
    -------
    sbs_avg_cost_train : float
        (Average) Cost of solving the training instances using the SBS.
    vbs_avg_cost_train : float
        (Average) Cost of solving the training instances using the VBS.
    sbs_avg_cost_test : float
        (Average) Cost of solving the test instances using the SBS.
    vbs_avg_cost_test : float
        (Average) Cost of solving the test instances using the VBS.
    """     
    # YOUR CODE HERE
    # Calculate the SBS for both train and test datasets

    # Calculate the average cost for each input

    return (calculate_sbs(train_performance_data), calculate_vbs(train_performance_data),
            calculate_sbs(test_performance_data), calculate_vbs(test_performance_data))


def evaluate_as_model(performance_data, predicted_algs, vbs_avg_cost, sbs_avg_cost):
    """Evaluate an AS model

    Parameters
    ----------
    performance_data : {ndarray} of shape (#instances, #algorithms)
        Cost of solving each instance in a given instance set by each algorithm.
    predicted_algs : 1-D array with #instances elements
        Index of the algorithm selected by the AS model for each instance. Algorithm index starts from 0.
    vbs_avg_cost : float
        (Average) Cost of solving the given instances using the VBS.
    sbs_avg_cost : float
        (Average) ost of solving the given instances using the SBS.

    Returns
    -------    
    avg_cost : float
        (Average) Cost of solving the given instances using the algorithms selected by the AS model.
    sbs_vbs_gap : float
        SBS-VBS gap of the AS model.
    """
    # Calculate the average cost of solving instances using the AS model
    l_avg_cost = []
    for i, row in enumerate(performance_data):
        l_avg_cost.append(row[predicted_algs[i]])
    avg_cost = np.mean(l_avg_cost)

    # Calculate the SBS-VBS gap of the AS model
    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)

    """
    # Plot a bar chart to visualize the predicted algorithms selected by the AS model
    unique_predicted_algs, counts = np.unique(predicted_algs, return_counts=True)


    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_predicted_algs, counts)
    plt.xlabel('Algorithm Index')
    plt.ylabel('Frequency')
    plt.title('Predicted Algorithms Selected by AS Model')
    plt.xticks(unique_predicted_algs, [f'Algorithm {i}' for i in unique_predicted_algs])
    plt.show()
    """
    return avg_cost, sbs_vbs_gap

def plot_model_gaps(model_data, learning_threshold):
    models = list(model_data.keys())
    training_gaps = [data[0] for data in model_data.values()]
    testing_gaps = [data[1] for data in model_data.values()]

    bar_width = 0.35
    index = np.arange(len(models))

    plt.bar(index, training_gaps, bar_width, label="Training Gap", align='center', color='skyblue')
    plt.bar(index + bar_width, testing_gaps, bar_width, label="Testing Gap", align='center', color='lightcoral')

    plt.xlabel("Models")
    plt.ylabel("Gap Values")
    plt.title("Training and Testing SBS-VBS Gaps")
    plt.xticks(index + bar_width / 2, models)


    plt.axhline(y=learning_threshold, color='darkgreen', linestyle='--', label="SBS-VBS Learning Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_costs(model_data, thresholds):
    models = list(model_data.keys())
    training_costs = [data[0] for data in model_data.values()]
    testing_costs = [data[1] for data in model_data.values()]

    bar_width = 0.35
    index = np.arange(len(models))

    plt.bar(index, training_costs, bar_width, label="Training Cost", align='center', color='skyblue')
    plt.bar(index + bar_width, testing_costs, bar_width, label="Testing Cost", align='center', color='lightcoral')

    plt.xlabel("Models")
    plt.ylabel("Cost Values")
    plt.title("Training and Testing Average Costs")
    plt.xticks(index + bar_width / 2, models)

    # Add the threshold lines with labels
    for key, value in thresholds.items():
        if key == "SBS Training":
            plt.axhline(y=value, color='navy',linestyle='--', label=key)
        elif key == "VBS Training":
            plt.axhline(y=value, color='navy', linestyle='solid', label=key)
        elif key == "SBS Testing" :
            plt.axhline(y=value, color='maroon',linestyle='--', label=key)
        elif key == "VBS Testing":
            plt.axhline(y=value, color='maroon',linestyle='solid', label=key)

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    plt.show()


def calculate_more_metrics(y_true, y_pred, model_type):
    """Calculate performance metrics based on model type (classification or regression).

    Parameters
    ----------
    y_true : list or numpy array
        Actual labels/targets.
    y_pred : list or numpy array
        Predicted labels/targets.
    model_type : str
        Type of model, must be either "classification" or "regression".

    Returns
    -------
    metrics : dict
        Dictionary containing performance metrics.
    """
    metrics = {}
    if model_type == "classification":
        # Calculate classification metrics
        # You can use different metrics depending on your preference
        # For example, accuracy, precision, recall, F1-score, etc.
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, average='weighted')
        metrics["Recall"] = recall_score(y_true, y_pred, average='weighted')
        metrics["F1-Score"] = f1_score(y_true, y_pred, average='weighted')

    elif model_type == "regression":
        # Calculate regression metrics
        metrics["Mean Absolute Error (MAE)"] = mean_absolute_error(y_true, y_pred)
        metrics["Mean Squared Error (MSE)"] = mean_squared_error(y_true, y_pred)
        # Add more regression metrics if needed
    # metrics["SBS-VBS Gap"] = sbs_vbs_gap
    return metrics


def plot_metrics(train_metrics, test_metrics, model_name):
    """Plot performance metrics.

    Parameters
    ----------
    train_metrics : dict
        Dictionary containing training performance metrics.
    test_metrics : dict
        Dictionary containing test performance metrics.
    """
    # Create a bar plot to compare train and test metrics
    metrics_names = list(train_metrics.keys())
    train_values = [train_metrics[key] for key in metrics_names]
    test_values = [test_metrics[key] for key in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, train_values, width, label='Train')
    ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_ylabel('Metrics')
    ax.set_title(f"Performance Metrics Linear: {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()

    plt.tight_layout()
    plt.show()
