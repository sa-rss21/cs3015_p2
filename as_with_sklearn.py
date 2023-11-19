import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer, average_precision_score, brier_score_loss, balanced_accuracy_score, \
    accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils import read_data, get_vbs_sbs, evaluate_as_model, calculate_vbs, calculate_sbs
from sklearn.pipeline import Pipeline
import pandas as pd


learning_gaps = []
start = 0

def sbs_vbs_gap_scorer(y_true, y_pred):
  vbs = calculate_vbs(y_true)
  sbs = calculate_sbs(y_true)
  cost, gap = evaluate_as_model(y_true, [np.argmin(row) for row in y_pred], vbs, sbs)
  print(gap, len(learning_gaps)//3)
  learning_gaps.append(gap)
  return gap


def train_and_evaluate_as_model(data_dir, model, model_type, use_scaler, param_grid=None):
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
    scorer = make_scorer(score_func=sbs_vbs_gap_scorer, greater_is_better=False)
    if use_scaler:
        pipe.steps.append(("scale", RobustScaler()))
        if model_type == "classification":
            pipe.steps.append(("scale2", MinMaxScaler()))
    pipe.steps.append(("model", model))
    mod = pipe

    if model_type == "classification":

        mod.fit(train_instance_features, algo_targets)
        predicted_algos = mod.predict(test_instance_features)
        predicted_algos_train = mod.predict(train_instance_features)
    elif model_type == "regression":
        if param_grid:
            grid = GridSearchCV(pipe, cv=3, param_grid=param_grid, scoring=scorer)
            grid.fit(train_instance_features, train_performance_data)
            print(grid.best_params_)
            mod = grid.best_estimator_
        else:
            mod.fit(train_instance_features, train_performance_data)
        predicted_algos = [np.argmin(row) for row in mod.predict(test_instance_features)]
        predicted_algos_train = [np.argmin(row) for row in mod.predict(train_instance_features)]

    if param_grid and model_type == "regression":
        plot = plt
        # Create a plot

        sorted_data = sorted(learning_gaps, reverse=True)
        plot.plot(sorted_data)

        # Add labels and title (optional)
        plot.xlabel('Searches')
        plot.ylabel('Gap')
        plot.yscale('log')
        plot.title('learning gap over time')
        results = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score']]#, 'rank_test_score']]
        results.sort_values('mean_test_score')
        print(results)
        # Specify the file path where you want to save the results
        output_file = '/grid_search_results_NR.csv'

        # Save the results to a CSV file
        results.to_csv(output_file, index=False)  # Set index=False to exclude the index column
    avg_cost_test, sbs_vbs_gap_test = evaluate_as_model(test_performance_data, predicted_algos,
                                                        vbs_avg_cost_test, sbs_avg_cost_test)

    avg_cost_train, sbs_vbs_gap_train = evaluate_as_model(train_performance_data, predicted_algos_train,
                                                          vbs_avg_cost_train, sbs_avg_cost_train)

    return avg_cost_train, sbs_vbs_gap_train, avg_cost_test, sbs_vbs_gap_test