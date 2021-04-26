import pandas as pd


def head_view(data: pd.DataFrame) -> pd.DataFrame:
	return data.head()


def tail_view(data: pd.DataFrame) -> pd.DataFrame:
	return data.tail()


def statistics_view(data: pd.DataFrame) -> pd.DataFrame:
	return data.describe()


def nan_view(data: pd.DataFrame) -> pd.DataFrame:
	tmp = pd.DataFrame(data.isna().sum(axis=0)).reset_index(level=0)
	tmp.columns = ['Column name', 'Number of NaN']
	return tmp
