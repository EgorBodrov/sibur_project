import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _fill_nans(data):
	return data.fillna(data.mean(), inplace=False)


def make_plot(data: pd.DataFrame):

	df = _fill_nans(data)

	fig = plt.figure(figsize=(10,5))
	cols = ['A_rate', 'B_rate']
	df.index = df.timestamp.astype("datetime64[ns]")
	means = df[cols].resample('M').mean()
	plt.plot(means, label=cols)
	plt.legend()
	return fig
	

def make_piechart(data: pd.DataFrame):
	df = data.loc[:, 'A_CH4':'A_C6H14']

	fig = plt.figure(figsize=(10,5))
	plt.pie(df.mean(), autopct='%1.1f%%', labels=df.columns)
	return fig


def make_correlation(data: pd.DataFrame):
	cor = data.loc[:,'A_rate':'A_C6H14'].corr()
	fig = sns.heatmap(cor).get_figure()
	return fig