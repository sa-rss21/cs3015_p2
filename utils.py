
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

    return avg_cost, sbs_vbs_gap


def plot_model_gaps(model_data, learning_threshold):
    models = list(model_data.keys())
    training_gaps = [data[0] for data in model_data.values()]
    testing_gaps = [data[1] for data in model_data.values()]

    bar_width = 0.35
    index = np.arange(len(models))
    plot = plt
    plot.bar(index, training_gaps, bar_width, label="Training Gap", align='center', color='skyblue')
    plot.bar(index + bar_width, testing_gaps, bar_width, label="Testing Gap", align='center', color='lightcoral')

    plot.xlabel("Models")
    plot.ylabel("Gap Values")
    plot.title("Training and Testing SBS-VBS Gaps")
    plot.xticks(index + bar_width / 2, models)

    plot.axhline(y=learning_threshold, color='darkgreen', linestyle='--', label="SBS-VBS Learning Threshold")
    plot.legend()
    plot.tight_layout()
    plot.show()
    plot.clf()


def plot_model_costs(model_data, thresholds):
    models = list(model_data.keys())
    training_costs = [data[0] for data in model_data.values()]
    testing_costs = [data[1] for data in model_data.values()]

    bar_width = 0.35
    index = np.arange(len(models))
    plot = plt
    plot.bar(index, training_costs, bar_width, label="Training Cost", align='center', color='skyblue')
    plot.bar(index + bar_width, testing_costs, bar_width, label="Testing Cost", align='center', color='lightcoral')

    plot.xlabel("Models")
    plot.ylabel("Cost Values")
    plot.title("Training and Testing Average Costs")
    plot.xticks(index + bar_width / 2, models)

    # Add the threshold lines with labels
    for key, value in thresholds.items():
        if key == "SBS Training":
            plot.axhline(y=value, color='navy',linestyle='--', label=key)
        elif key == "VBS Training":
            plot.axhline(y=value, color='navy', linestyle='solid', label=key)
        elif key == "SBS Testing" :
            plot.axhline(y=value, color='maroon',linestyle='--', label=key)
        elif key == "VBS Testing":
            plot.axhline(y=value, color='maroon',linestyle='solid', label=key)

    plot.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plot.tight_layout()
    plot.show()
    plot.clf()
