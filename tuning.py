
import os

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier

script_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.append(script_dir)

from sklearn.linear_model import LogisticRegression, LinearRegression

from utils import read_data, get_vbs_sbs, evaluate_as_model, plot_model_gaps, plot_model_costs
from as_with_sklearn import train_and_evaluate_as_model

linear_regression_param_grid = {
    'model__fit_intercept': [True, False],
    'model__copy_X': [True, False],
    'model__n_jobs': range(2, 6),
    'model__positive': [True, False]
}
linear_classification_param_grid = {
    'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'model__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #'model__max_iter': [100, 500, 1000],
    #'model__multi_class': ['auto', 'ovr', 'multinomial'],
    #'scale__with_mean': [True, False],
    #'scale__with_std': [True, False]
}
neural_classification_param_grid = {
    'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001, 0.01],
    'model__solver': ['adam', 'sgd'],
    'model__learning_rate': ['constant', 'adaptive'],
    'model__max_iter': [500, 1000, 2000],
    'model__early_stopping': [True, False],
    'model__validation_fraction': [0.1, 0.2, 0.3],
    'scale__with_mean': [True, False],
    'scale__with_std': [True, False]
}
neural_regression_param_grid = {
    'model__hidden_layer_sizes': [(50,), (100,), (200,), (50, 50), (100, 50)],
    'model__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'model__solver': ['lbfgs', 'adam'],
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],
    'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'model__max_iter': [500],  # Set the max_iter value to the one that works well
}
# read data
data_dir = f"{script_dir}/data/"
train_performance_data, train_instance_features, test_performance_data, test_instance_features = read_data(data_dir)
model_data = {}
model_data_costs = {}

# get vbs and sbs
sbs_avg_cost_train, vbs_avg_cost_train, sbs_avg_cost_test, vbs_avg_cost_test = get_vbs_sbs(train_performance_data, test_performance_data)
print(f"sbs_train:{sbs_avg_cost_train:10.2f}, vbs_train:{vbs_avg_cost_train:10.2f}")
print(f"sbs_test :{sbs_avg_cost_test:10.2f}, vbs_test :{vbs_avg_cost_test:10.2f}")


model = MLPRegressor()
avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test = (
    train_and_evaluate_as_model(data_dir, model, "regression", True, neural_regression_param_grid))
model_data["Neural\nRegression"] = (sbs_vbs_gap_train, sbs_vbs_gap_test)
model_data_costs["Neural\nRegression"] = (avg_cost_train, avg_cost_test)


# Plot the performance metrics
plot_model_gaps(model_data, 1)
plot_model_costs(model_data_costs, {"SBS Training":sbs_avg_cost_train, "SBS Testing":sbs_avg_cost_test,
                              "VBS Testing": vbs_avg_cost_test, "VBS Training":vbs_avg_cost_train})
