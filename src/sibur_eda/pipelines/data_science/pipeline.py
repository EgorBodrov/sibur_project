from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs):
	return Pipeline(
        [
            node(
                func=data_fillnans,
                inputs="sibur",
                outputs="fillnans",
                name="data_fillnans",
            ),
            node(
                func=model_training_args,
                inputs="fillnans",
                outputs="args",
                name="model_args",
            ),
            node(
                func=train_A_CH4,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_CH4', 'X_test_A_CH4', 'y_test_A_CH4'],
                name="train_results_A_CH4",
            ),
            node(
                func=evaluate_A_CH4,
                inputs=['regressor_A_CH4', 'X_test_A_CH4', 'y_test_A_CH4'],
                outputs='result_A_CH4',
                name="evaluate_results_A_CH4",
            ),
            node(
                func=train_A_C2H6,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_C2H6', 'X_test_A_C2H6', 'y_test_A_C2H6'],
                name="train_results_A_C2H6",
            ),
            node(
                func=evaluate_A_C2H6,
                inputs=['regressor_A_C2H6', 'X_test_A_C2H6', 'y_test_A_C2H6'],
                outputs='result_A_C2H6',
                name="evaluate_results_A_C2H6",
            ),
            node(
                func=train_A_C3H8,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_C3H8', 'X_test_A_C3H8', 'y_test_A_C3H8'],
                name="train_results_A_C3H8",
            ),
            node(
                func=evaluate_A_C3H8,
                inputs=['regressor_A_C3H8', 'X_test_A_C3H8', 'y_test_A_C3H8'],
                outputs='result_A_C3H8',
                name="evaluate_results_A_C3H8",
            ),
            node(
                func=train_A_iC4H10,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_iC4H10', 'X_test_A_iC4H10', 'y_test_A_iC4H10'],
                name="train_results_A_iC4H10",
            ),
            node(
                func=evaluate_A_iC4H10,
                inputs=['regressor_A_iC4H10', 'X_test_A_iC4H10', 'y_test_A_iC4H10'],
                outputs='result_A_iC4H10',
                name="evaluate_results_A_iC4H10",
            ),
            node(
                func=train_A_nC4H10,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_nC4H10', 'X_test_A_nC4H10', 'y_test_A_nC4H10'],
                name="train_results_A_nC4H10",
            ),
            node(
                func=evaluate_A_nC4H10,
                inputs=['regressor_A_nC4H10', 'X_test_A_nC4H10', 'y_test_A_nC4H10'],
                outputs='result_A_nC4H10',
                name="evaluate_results_A_nC4H10",
            ),
            node(
                func=train_A_iC5H12,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_iC5H12', 'X_test_A_iC5H12', 'y_test_A_iC5H12'],
                name="train_results_A_iC5H12",
            ),
            node(
                func=evaluate_A_iC5H12,
                inputs=['regressor_A_iC5H12', 'X_test_A_iC5H12', 'y_test_A_iC5H12'],
                outputs='result_A_iC5H12',
                name="evaluate_results_A_iC5H12",
            ),
            node(
                func=train_A_nC5H12,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_nC5H12', 'X_test_A_nC5H12', 'y_test_A_nC5H12'],
                name="train_results_A_nC5H12",
            ),
            node(
                func=evaluate_A_nC5H12,
                inputs=['regressor_A_nC5H12', 'X_test_A_nC5H12', 'y_test_A_nC5H12'],
                outputs='result_A_nC5H12',
                name="evaluate_results_A_nC5H12",
            ),
            node(
                func=train_A_C6H14,
                inputs=['args', 'params:model_params'],
                outputs=['regressor_A_C6H14', 'X_test_A_C6H14', 'y_test_A_C6H14'],
                name="train_results_A_C6H14",
            ),
            node(
                func=evaluate_A_C6H14,
                inputs=['regressor_A_C6H14', 'X_test_A_C6H14', 'y_test_A_C6H14'],
                outputs='result_A_C6H14',
                name="evaluate_results_A_C6H14",
            ),
        ]
    )