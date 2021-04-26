import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Tuple, Dict


def clean_data(train, test, column):
    if column == 'A_CH4':
        a, b = 0.05, 0.33
    elif column == 'A_C2H6':
        a, b = 2.3, 6
    elif column == 'A_C3H8':
        a, b = 33, 41.5
    elif column == 'A_iC4H10':
        a, b = 16, 19
    elif column == 'A_nC4H10':
        a, b = 22.5, 27
    elif column == 'A_iC5H12':
        a, b = 3.9, 6.5
    elif column == 'A_nC5H12':
        a, b = 3, 6.5
    elif column == 'A_C6H14':
        a, b = 2.5, 7.5

    tmp = [el for el in zip(train, test) if el[1] > a if el[1] < b]
    train = [el[0] for el in tmp]
    test = [el[1] for el in tmp]
    return train, test


def data_fillnans(data: pd.DataFrame) -> pd.DataFrame:
	data = data.fillna(data.mean(axis=0))
	return data


def model_training_args(data: pd.DataFrame) -> pd.DataFrame:
	useful_cols = [elem for elem in data.columns.to_list() if
               elem not in ['timestamp', 'A_rate', 'B_rate']]

	args = pd.DataFrame(index=useful_cols, columns=['X', 'y'])
	for column in useful_cols:
		X = [[elem1, elem2] for elem1, elem2 in zip(data['A_rate'], data['B_rate'])]
		y = data[column].to_list()

		X, y = clean_data(X, y, column)
		args.loc[column] = [X, y]

	return args


def train_A_CH4(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_CH4', 'X']
	y = args.loc['A_CH4', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_CH4(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_CH4'])
	result.loc['A_CH4'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_CH4'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_C2H6(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_C2H6', 'X']
	y = args.loc['A_C2H6', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_C2H6(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_C2H6'])
	result.loc['A_C2H6'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_C2H6'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_C3H8(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_C3H8', 'X']
	y = args.loc['A_C3H8', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_C3H8(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_C3H8'])
	result.loc['A_C3H8'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_C3H8'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_iC4H10(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_iC4H10', 'X']
	y = args.loc['A_iC4H10', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_iC4H10(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_iC4H10'])
	result.loc['A_iC4H10'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_iC4H10'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_nC4H10(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_nC4H10', 'X']
	y = args.loc['A_nC4H10', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_nC4H10(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_nC4H10'])
	result.loc['A_nC4H10'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_nC4H10'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_iC5H12(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_iC5H12', 'X']
	y = args.loc['A_iC5H12', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_iC5H12(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_iC5H12'])
	result.loc['A_iC5H12'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_iC5H12'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_nC5H12(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_nC5H12', 'X']
	y = args.loc['A_nC5H12', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_nC5H12(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_nC5H12'])
	result.loc['A_nC5H12'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_nC5H12'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result


def train_A_C6H14(args: pd.DataFrame, params: Dict) -> Dict:
	X = args.loc['A_C6H14', 'X']
	y = args.loc['A_C6H14', 'y']

	X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']) 

	regressor = Lasso()
	regressor.fit(X_train, y_train)

	return regressor, X_test, y_test


def evaluate_A_C6H14(regressor: Lasso, X_test: list, y_test: list) -> pd.DataFrame:
	predicted = regressor.predict(X_test)

	result = pd.DataFrame(columns=['Actual', 'Predicted'], index=['A_C6H14'])
	result.loc['A_C6H14'] = [y_test, predicted]

	logger = logging.getLogger(__name__)
	logger.info(f"\nFor 'A_C6H14'\n \
            	r2_score = {r2_score(y_test, predicted)}\n \
            	MAE = {mean_absolute_error(y_test, predicted)}\n \
            	RMSE = {mean_squared_error(y_test, predicted)}\n")

	return result
