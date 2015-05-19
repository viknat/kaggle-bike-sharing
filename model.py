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
from sklearn.metrics import mean_squared_error 

def read_data():
	df = pd.read_csv('train.csv')
	return df

def fit_model(X,y,model):
	#model = model()
	model.fit(X,y)
	return model

def clean_data(df):
	df['datetime'] = df['datetime'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
	df['hour'] = df['datetime'].apply(lambda date: date.hour)

	daylight_condition = lambda hour: hour > 6 and hour < 21

	#df['daylight'] = df['hour'].apply(lambda hour: daylight_condition(hour))
	# df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 10, 15, 20])
	# tmp = pd.get_dummies(df['time_of_day'])
	# df = df.merge(tmp, left_index=True, right_index=True, copy=False)
	
	# df = df.drop(['hour', '(0, 6]', 'time_of_day'], axis=1)

	tmp = pd.get_dummies(df['hour'])
	df = df.merge(tmp, left_index=True, right_index=True, copy=False)
	df = df.drop(['hour', 0] , axis=1)

	df['day_of_week'] = df['datetime'].apply(lambda date: date.dayofweek)

	tmp = pd.get_dummies(df['day_of_week'])
	tmp.columns = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
	df = df.merge(tmp, left_index=True, right_index=True, copy=False)
	df = df.drop(['workingday', 'day_of_week', 'Sunday'] , axis=1)

	df['year'] = df['datetime'].apply(lambda date: date.year)
	df['year'] = df['year'] == 2012

	season_dummies = pd.get_dummies(df['season'])
	season_dummies.columns = ['spring','summer','autumn','winter']
	season_dummies.drop('winter', axis=1, inplace=True)
	df = df.merge(season_dummies, left_index=True, right_index=True, copy=False)
	df = df.drop(['season'], axis=1)
	try:
		df = df.drop(['casual', 'registered'], axis=1)
	except ValueError:
		pass

	weather_dummies = pd.get_dummies(df['weather'])
	weather_dummies.columns = ['weather1', 'weather2', 'weather3', 'weather4']
	#weather_dummies.drop('weather4', axis=1, inplace=True)
	df = df.merge(weather_dummies, left_index=True, right_index=True, copy=False)
	df = df.drop(['weather', 'weather1'], axis=1)

	df = df.drop(['holiday', 'temp'], axis=1)
	df = df.drop('datetime', axis=1)
	if 'count' in df.columns.values:
		df['count'] = df['count'].apply(lambda c: np.log(c+1))
		#print (df.columns)

	return df





df = read_data()
df = clean_data(df)

print (df.columns)

y = df.pop('count').values
X = df.values

# X_train, X_test, y_train, y_test = train_test_split \
# (X, y, test_size = 0.3)


def rmsle(y_pred, y_test):

	# print (y_pred)
	# print (y_test)

	# log_pred = np.log((y_pred)+1)
	# log_test = np.log(y_test+1) 
	# return np.sum(np.square((log_pred - log_test)))/len(y_pred)
	return np.sum(np.square((y_pred - y_test)))/len(y_pred)



def score_rmsle(model):	
	y_pred = model.predict(X_test)

	#print (type(y_pred))
	# for yex in [y_test, y_pred]:
	# 	#yex = np.exp((yex.astype(float))) - 1
	# 	yex = np.array([int(np.around(y)) for y in yex])
	#yex[yex < 0] = 0

	print (y_pred)
	print (y_test)

	score = rmsle(y_pred, y_test)

	return (model.__class__.__name__, score)

def print_feature_importances(forest, columns):
	print ("Feature importances: ")
	for col, feat in zip(columns, forest.feature_importances_):
		print (col, feat)

# linear_model = fit_model(X_train,y_train, LinearRegression())
# forest = fit_model(X_train, y_train, RandomForestClassifier())

#print (linear_model.score(X_train, y_train))
#print (forest.score(X_train, y_train))

linear_model = fit_model(X, y, LinearRegression())

# print (score_rmsle(linear_model))
# print (score_rmsle(forest))

X_results = pd.read_csv('test.csv')
dates = X_results['datetime'].values

X_results = clean_data(X_results)



X_np = X_results.values
with open('results.csv', 'w', newline='') as output:
	a = csv.writer(output, delimiter=',')
	y_results = linear_model.predict(X_np)
	y_results = np.exp((y_results.astype(float))) - 1

	y_results = np.array([int(np.around(y)) for y in y_results])
	a.writerows([['datetime', 'count']])
	for date, c in zip(dates, y_results):
		a.writerows([[date, c]])
