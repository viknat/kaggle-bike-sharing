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

class KaggleBikeShare():

	def __init__(self, train_fname, test_fname):
		self.training_set = pd.read_csv(train_fname)
		self.test_set = pd.read_csv(test_fname)

	def get_and_merge_dummies(self, df, column, dummy_column_names=None):
		dummy_df = pd.get_dummies(df[column])
		if dummy_column_names:
			dummy_df.columns = dummy_column_names
		dummy_df.drop(dummy_df.columns[0], axis=1, inplace=True)
		df.drop(column, axis=1, inplace=True)
		df = df.merge(dummy_df, left_index=True, right_index=True, copy=False)
		return df

	def engineer_features(self, df):
		df['datetime'] = df['datetime'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
		df['day_of_week'] = df['datetime'].apply(lambda date: date.dayofweek)
		df['year'] = df['year'].apply(lambda date: date.year)
		
		df = self.get_and_merge_dummies(df, 'season', ['spring', 'summer', 'fall', 'winter'])
		df = self.get_and_merge_dummies(df, 'weather')
		df = self.get_and_merge_dummies(df, 'day_of_week', ['Sunday', 'Monday', 'Tuesday' \
										'Wednesday', 'Thursday', 'Friday', 'Saturday'])
		df = self.get_and_merge_dummies(df, 'year', lambda date: date.year)
		return df


if __name__ == '__main__':
	bike_share_object = KaggleBikeShare(train_fname='train.csv', test_fname='test.csv')
	df = bike_share_object.engineer_features(bike_share_object.training_set)

