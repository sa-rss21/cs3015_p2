
# THIS SCRIPT SHOULD NOT BE MODIFIED. 
# YOU MUST MAKE SURE THIS SCRIPT CAN RUN PROPERLY WITH YOUR SUBMITTED CODE.

# path to current script's directory
import os
script_dir = os.path.abspath(os.path.dirname(__file__))

# add path to current script's directory to Python's path
import sys
sys.path.append(script_dir)

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import read_data, get_vbs_sbs, evaluate_as_model
from as_with_sklearn import train_and_evaluate_as_model

# read data
data_dir = f"{script_dir}/data/"
train_performance_data, train_instance_features, test_performance_data, test_instance_features = read_data(data_dir)

# get vbs and sbs
sbs_avg_cost_train, vbs_avg_cost_train, sbs_avg_cost_test, vbs_avg_cost_test = get_vbs_sbs(train_performance_data, test_performance_data)
print(f"sbs_train:{sbs_avg_cost_train:10.2f}, vbs_train:{vbs_avg_cost_train:10.2f}")
print(f"sbs_test :{sbs_avg_cost_test:10.2f}, vbs_test :{vbs_avg_cost_test:10.2f}")

# evaluate performance of a (random) AS model
print("\nA random AS model:")
predicted_algs = np.random.randint(test_performance_data.shape[1], size=test_performance_data.shape[0])
avg_cost_test, sbs_vbs_gap_test = evaluate_as_model(test_performance_data, predicted_algs, vbs_avg_cost_test, sbs_avg_cost_test)
print(f"avg_cost_test:{avg_cost_test:10.2f}, sbs_vbs_gap_test:{sbs_vbs_gap_test:3.2f}")

# train and evaluate an AS model
print("\nA linear classification AS model:")
model = LogisticRegression()
avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test = train_and_evaluate_as_model(data_dir, model, "classification", False)
print(f"avg_cost_train:{avg_cost_train:10.2f}, sbs_vbs_gap_train:{sbs_vbs_gap_train:5.2f}")
print(f"avg_cost_test :{avg_cost_test:10.2f}, sbs_vbs_gap_test :{sbs_vbs_gap_test:5.2f}")