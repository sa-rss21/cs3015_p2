import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import read_data, get_vbs_sbs, evaluate_as_model, plot_model_gaps, calculate_more_metrics, plot_metrics
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

def train_and_evaluate_as_model(data_dir, model, model_type, use_scaler):
    """Train and evaluate an AS model

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset folder.
    model: sklearn.base.BaseEstimator
        A scikit-learn model.
    model_type: str
        Type of model, must be either "classification" or "regression".
    use_scaler: bool
        Whether the instance features are normalised (using scikit-learn StandardScaler).

    Returns
    -------
    avg_cost_train : float
        (Average) Cost of solving the training instances using the model.
    sbs_vbs_gap_train : float
        (Average) SBS-VBS gap on the training instances.
    avg_cost_test : float
        (Average) Cost of solving the test instances using the model.
    sbs_vbs_gap_test : float
        (Average) SBS-VBS gap on the test instances.
    """
    assert model_type in ["classification", "regression"]


    # data collection
    train_performance_data, train_instance_features, test_performance_data, test_instance_features = read_data(data_dir)
    sbs_avg_cost_train, vbs_avg_cost_train, sbs_avg_cost_test, vbs_avg_cost_test = get_vbs_sbs(train_performance_data,
                                                                                               test_performance_data)

    algo_targets = [np.argmin(row) for row in train_performance_data]

    # model pipeline
    pipe = Pipeline([])
    if use_scaler:
        pipe.steps.append(("scale", StandardScaler()))
    pipe.steps.append(("model", model))

    if model_type == "classification":

        pipe.fit(train_instance_features, algo_targets)
        predicted_algos = pipe.predict(test_instance_features)
        predicted_algos_train = pipe.predict(train_instance_features)
    elif model_type == "regression":
        pipe.fit(train_instance_features, train_performance_data)
        predicted_algos = [np.argmin(row) for row in pipe.predict(test_instance_features)]
        predicted_algos_train = [np.argmin(row) for row in pipe.predict(train_instance_features)]

    avg_cost_test, sbs_vbs_gap_test = evaluate_as_model(test_performance_data, predicted_algos,
                                                        vbs_avg_cost_test, sbs_avg_cost_test)

    avg_cost_train, sbs_vbs_gap_train = evaluate_as_model(train_performance_data, predicted_algos_train,
                                                        vbs_avg_cost_train, sbs_avg_cost_train)

    return avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test


