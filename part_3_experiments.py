import os

from sklearn.neural_network import MLPRegressor, MLPClassifier

script_dir = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.append(script_dir)

from sklearn.linear_model import LogisticRegression, LinearRegression

from utils import read_data, get_vbs_sbs, evaluate_as_model, plot_model_gaps, plot_model_costs
from as_with_sklearn import train_and_evaluate_as_model


# read data
data_dir = f"{script_dir}/data/"
train_performance_data, train_instance_features, test_performance_data, test_instance_features = read_data(data_dir)
model_data = {}
model_data_costs = {}

# get vbs and sbs
sbs_avg_cost_train, vbs_avg_cost_train, sbs_avg_cost_test, vbs_avg_cost_test = get_vbs_sbs(train_performance_data, test_performance_data)
print(f"sbs_train:{sbs_avg_cost_train:10.2f}, vbs_train:{vbs_avg_cost_train:10.2f}")
print(f"sbs_test :{sbs_avg_cost_test:10.2f}, vbs_test :{vbs_avg_cost_test:10.2f}")

model = MLPClassifier(max_iter=500, hidden_layer_sizes=(150, 100, 50, 25))
avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test = train_and_evaluate_as_model(data_dir, model, "classification", True)
model_data["Neural\nClassification\nOptimized"] = (sbs_vbs_gap_train, sbs_vbs_gap_test)
model_data_costs["Neural\nClassification\nOptimized"] = (avg_cost_train, avg_cost_test)

# Plot the performance metrics
plot_model_gaps(model_data, 1)
plot_model_costs(model_data_costs, {"SBS Training":sbs_avg_cost_train, "SBS Testing":sbs_avg_cost_test,
                              "VBS Testing": vbs_avg_cost_test, "VBS Training":vbs_avg_cost_train})