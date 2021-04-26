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


def make_A_CH4(result: pd.DataFrame):
	x = result.loc['A_CH4', 'Actual']
	y = result.loc['A_CH4', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_C2H6(result: pd.DataFrame):
	x = result.loc['A_C2H6', 'Actual']
	y = result.loc['A_C2H6', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_C3H8(result: pd.DataFrame):
	x = result.loc['A_C3H8', 'Actual']
	y = result.loc['A_C3H8', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_iC4H10(result: pd.DataFrame):
	x = result.loc['A_iC4H10', 'Actual']
	y = result.loc['A_iC4H10', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_nC4H10(result: pd.DataFrame):
	x = result.loc['A_nC4H10', 'Actual']
	y = result.loc['A_nC4H10', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_iC5H12(result: pd.DataFrame):
	x = result.loc['A_iC5H12', 'Actual']
	y = result.loc['A_iC5H12', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_nC5H12(result: pd.DataFrame):
	x = result.loc['A_nC5H12', 'Actual']
	y = result.loc['A_nC5H12', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig


def make_A_C6H14(result: pd.DataFrame):
	x = result.loc['A_C6H14', 'Actual']
	y = result.loc['A_C6H14', 'Predicted']

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
    
	return fig

