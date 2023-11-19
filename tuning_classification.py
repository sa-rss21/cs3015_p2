
import os
from sklearn.neural_network import MLPClassifier

script_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.append(script_dir)

from utils import read_data, get_vbs_sbs, plot_model_gaps, plot_model_costs
from as_with_sklearn import train_and_evaluate_as_model



# read data
data_dir = f"{script_dir}/data/"
train_performance_data, train_instance_features, test_performance_data, test_instance_features = read_data(data_dir)
model_data = {}
model_data_costs = {}

# get vbs and sbs
sbs_avg_cost_train, vbs_avg_cost_train, sbs_avg_cost_test, vbs_avg_cost_test = get_vbs_sbs(train_performance_data, test_performance_data)

param_grid = {
    'hidden_layer_sizes': [(100, 50, 25), (100, 100, 50), (100, 50), (200, 100, 50)],
    'activation': ['relu', 'logistic', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'max_iter': [200, 500, 1000],
    'alpha': [0.0001, 0.001, 0.01, .1, 1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [.0001, .001, .01, .1, 1, 10],
    'power_t': [.1, .25, .5, .75, 1],
    'random_state': [42] , # You can change this to any specific random state value you prefer
    'tol': [.0001, .001, .01, .1, 1]
}
cost_data = []
gap_data = []
param = 'learning_rate_init'
for val in param_grid[param]:
    model = MLPClassifier(max_iter=300, hidden_layer_sizes=(200, 100, 50), solver='lbfgs',
                          learning_rate_init=val, learning_rate='adaptive', activation='tanh')
    avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test = (
        train_and_evaluate_as_model(data_dir, model, "classification", True))
    model_data[f"{val}"] = (sbs_vbs_gap_train, sbs_vbs_gap_test)
    model_data_costs[f"{val}"] = (avg_cost_train, avg_cost_test)


# Plot the performance metrics
plot_model_gaps(model_data, 1)
plot_model_costs(model_data_costs, {"SBS Training":sbs_avg_cost_train, "SBS Testing":sbs_avg_cost_test,
                              "VBS Testing": vbs_avg_cost_test, "VBS Training":vbs_avg_cost_train})