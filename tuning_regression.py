
import os

from sklearn.neural_network import MLPRegressor

script_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.append(script_dir)

from sklearn.linear_model import LinearRegression

from utils import read_data, get_vbs_sbs, plot_model_gaps, plot_model_costs
from as_with_sklearn import train_and_evaluate_as_model

linear_regression_param_grid = {
    'model__fit_intercept': [True, False],
    'model__copy_X': [True, False],
    'model__positive': [True, False]
}

neural_regression_param_grid = {
    'model__hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100, 50)],
    'model__activation': ['identity', 'relu', 'tanh'],
    'model__solver': ['lbfgs', 'adam'],
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],
    'model__learning_rate': ['constant', 'adaptive'],
    'model__max_iter': [200, 500],  # Set the max_iter value to the one that works well
}

# read data
data_dir = f"{script_dir}/data/"
train_performance_data, train_instance_features, test_performance_data, test_instance_features = read_data(data_dir)
model_data = {}
model_data_costs = {}

# get vbs and sbs
sbs_avg_cost_train, vbs_avg_cost_train, sbs_avg_cost_test, vbs_avg_cost_test = get_vbs_sbs(train_performance_data, test_performance_data)

# Perform exhaustive grid searches on both regression models
model = MLPRegressor()
avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test = (
    train_and_evaluate_as_model(data_dir, model, "regression", True, neural_regression_param_grid))
model_data["Neural\nRegression\nOptimized"] = (sbs_vbs_gap_train, sbs_vbs_gap_test)
model_data_costs["Neural\nRegression\nOptimized"] = (avg_cost_train, avg_cost_test)

model = LinearRegression()
avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test = (
    train_and_evaluate_as_model(data_dir, model, "regression", True, linear_regression_param_grid))
model_data["Linear\nRegression\nOptimized"] = (sbs_vbs_gap_train, sbs_vbs_gap_test)
model_data_costs["Linear\nRegression\nOptimized"] = (avg_cost_train, avg_cost_test)

# Plot the performance metrics
plot_model_gaps(model_data, 1)
plot_model_costs(model_data_costs, {"SBS Training":sbs_avg_cost_train, "SBS Testing":sbs_avg_cost_test,
                              "VBS Testing": vbs_avg_cost_test, "VBS Training":vbs_avg_cost_train})
