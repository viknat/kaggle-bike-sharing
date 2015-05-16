import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv
import math
import datetime 

def read_data():
	df = pd.read_csv('train.csv')
	'''pd.scatter_matrix(df);
	plt.show()'''
	return df

def fit_model(X,y,model):
	model = model()
	model.fit(X,y)
	return model

def clean_data(df):
	df['datetime'] = df['datetime'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
	df['hour'] = df['datetime'].apply(lambda date: date.hour)

	daylight_condition = lambda hour: hour > 6 and hour < 21

	df['daylight'] = df['hour'].apply(lambda hour: daylight_condition(hour))

	season_dummies = pd.get_dummies(df['season'])
	season_dummies.columns = ['spring','summer','autumn','winter']
	season_dummies.drop('winter', axis=1, inplace=True)
	df = df.merge(season_dummies, left_index=True, right_index=True, copy=False)
	df = df.drop('season', axis=1)
	try:
		df = df.drop(['casual', 'registered'], axis=1)
	except ValueError:
		pass

	weather_dummies = pd.get_dummies(df['weather'])
	weather_dummies.columns = ['weather1', 'weather2', 'weather3', 'weather4']
	weather_dummies.drop('weather4', axis=1, inplace=True)
	df = df.merge(weather_dummies, left_index=True, right_index=True, copy=False)
	df = df.drop(['weather'], axis=1)

	df = df.drop(['holiday', 'temp', 'hour'], axis=1)
	df = df.drop('datetime', axis=1)
	if 'count' in df.columns.values:
		df['count'] = df['count'].apply(lambda c: np.log(c+1))
		#print (df.columns)

	return df





df = read_data()
df = clean_data(df)

y = df.pop('count').values
X = df.values

X_train, X_test, y_train, y_test = train_test_split \
(X, y, test_size = 0.3)


def rmsle(y_pred, y_test):

	log_pred = np.log((y_pred)+1)
	log_test = np.log(y_test+1) 
	return np.sum(np.square((log_pred - log_test)))/len(y_pred)

def score_rmsle(model):
	y_pred = model.predict(X_test)
	print (type(y_pred))
	y_pred = np.exp((y_pred.astype(float))) - 1
	y_pred = np.array([int(np.around(y)) for y in y_pred])
	y_pred[y_pred < 0] = 0

	score = rmsle(y_pred, y_test)

	return (model.__class__.__name__, score)

linear_model = fit_model(X_train,y_train, LinearRegression)
forest = fit_model(X_train, y_train, RandomForestClassifier)

print (score_rmsle(linear_model))
print (score_rmsle(forest))
#print (list(zip(df.columns, forest.feature_importances_)))

X_results = pd.read_csv('test.csv')
dates = X_results['datetime'].values

X_results = clean_data(X_results)

print (X_results.columns.values)
X_np = X_results.values
with open('results.csv', 'w', newline='') as output:
	a = csv.writer(output, delimiter=',')
	y_results = linear_model.predict(X_np)
	y_results = np.exp((y_results.astype(float))) - 1
	a.writerows([['datetime', 'count']])
	for date, c in zip(dates, y_results):
		a.writerows([[date, c]])
