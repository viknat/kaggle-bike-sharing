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
import ipdb

class KaggleBikeShare():

	def __init__(self, train_fname, test_fname, target_name, models):
		self.training_set = pd.read_csv(train_fname)
		self.test_set = pd.read_csv(test_fname)
		self.models = models
		self.target_name = target_name

	def get_and_merge_dummies(self, df, column, dummy_column_names=None):
		'''
		Replaces a categorical variable with dummy variables
		INPUT: Pandas DataFrame, column name(to be dummified), names of dummy columns (optional)
		OUTPUT: DataFrame with replaced columns
		'''

		dummy_df = pd.get_dummies(df[column])
		if dummy_column_names:
			dummy_df.columns = dummy_column_names
		# Need to delete original column and one dummy column (preserves independence)
		dummy_df.drop(dummy_df.columns[0], axis=1, inplace=True)
		df.drop(column, axis=1, inplace=True)
		df = df.merge(dummy_df, left_index=True, right_index=True, copy=False)
		return df

	def engineer_features(self, df):
		'''
		Prepares the data for ML algorithms
		INPUT: Pandas DataFrame
		OUTPUT: Cleaned and feature-engineered DataFrame
		'''

		df['datetime'] = df['datetime'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
		df['day_of_week'] = df['datetime'].apply(lambda date: date.dayofweek)
		df['year'] = df['datetime'].apply(lambda date: date.year)
		df['hour'] = df['datetime'].apply(lambda date: date.hour)
		df['year'] = df['year'] == 2012
		
		#df = self.get_and_merge_dummies(df, 'season', ['spring', 'summer', 'fall', 'winter'])
		df['season'] = df['season'].map({2:1, 3:2, 4:3, 1:4})
		df['conditions'] = df['season'] + df['weather']
		df['conditions'] = df['conditions'] > 4
		print (df.head())
		df.drop(['season', 'weather'], axis=1, inplace=True)
		#df = self.get_and_merge_dummies(df, 'weather', ['good', 'fair', 'bad', 'harsh'])
		df = self.get_and_merge_dummies(df, 'day_of_week', ['Sunday', 'Monday', 'Tuesday', \
										'Wednesday', 'Thursday', 'Friday', 'Saturday'])
		df = self.get_and_merge_dummies(df, 'hour')


		df.drop(['atemp', 'holiday'], axis=1, inplace=True)
		if 'count' in df.columns: # If this is the training set
			df.drop(['casual', 'registered'], axis=1, inplace=True)
			df['count'] = df['count'].apply(lambda c: np.log(c+1))
		df.drop('datetime', axis=1, inplace=True)
		print (df.columns)
		return df

	def split_data_train_test(self, test_size=0.3):
		y = self.training_set.pop(self.target_name).values
		X = self.training_set.values

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
				X,y,test_size=test_size)

	def rmsle(self, y_pred, y_test):
		'''
		Returns root mean squared log error between the predicted and actual y 
		Note that the y's are already logs of the count
		'''
		pairwise_diff_square = np.square(y_pred - y_test)
		return np.sqrt((np.sum(pairwise_diff_square))/len(y_pred))

	def fit_and_score_model(self, model):
		model.fit(self.X_train, self.y_train)
		y_pred = model.predict(self.X_test)
		score = self.rmsle(y_pred, self.y_test)

		return (model.__class__.__name__, score)

	def run_models(self, cross_val=True, model=LinearRegression()):
		self.training_set = self.engineer_features(self.training_set)

		if cross_val:
			self.split_data_train_test(test_size=0.3)
			for model in self.models:
				print (self.fit_and_score_model(model))

		else:
			self.split_data_train_test(test_size=0.1)
			model.fit(self.X_train, self.y_train)
			print (self.X_train)
			print (self.X_train.shape)
			self.create_submission(model)

	def create_submission(self, fitted_model):
		dates = self.test_set['datetime'].values
		df = self.engineer_features(self.test_set) # Test set must match training
		#print (df.head())		
		print (dates)
		X_submit = df.values
		with open('results.csv', 'w', newline='') as output:
			a = csv.writer(output, delimiter=',')
			y_submit = fitted_model.predict(X_submit)
			y_submit = np.exp((y_submit.astype(float))) - 1

			y_submit = np.array([int(np.around(y)) for y in y_submit])
			a.writerows([['datetime', 'count']])
			for date, c in zip(dates, y_submit):
				a.writerows([[date, c]])

if __name__ == '__main__':
	models = [LinearRegression(), RandomForestClassifier()]
	bike_share_object = KaggleBikeShare(train_fname='train.csv', test_fname='test.csv', \
		target_name='count', models=models)
	bike_share_object.run_models(cross_val=True)



