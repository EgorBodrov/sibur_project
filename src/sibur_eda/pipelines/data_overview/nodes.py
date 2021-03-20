import pandas as pd

def head_view(data: pd.DataFrame) -> pd.DataFrame:
	return data.head()


def tail_view(data: pd.DataFrame) -> pd.DataFrame:
	return data.tail()


def statistics_view(data: pd.DataFrame) -> pd.DataFrame:
	return data.describe()

