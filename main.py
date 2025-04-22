import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import os
import joblib

#membagi kelas menjadi datahandler dan model handler

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    def drop_missing_data(self):
        self.data = self.data.dropna().reset_index(drop=True)
    def drop_column(self, column_name):
        if column_name in self.data.columns:
          self.data = self.data.drop(column_name, axis=1)
    def label_encode(self, column_name):
        label_encoder = preprocessing.LabelEncoder()
        self.data[column_name] = label_encoder.fit_transform(self.data[column_name])
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
        

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def one_hot_encode(self, columns_to_encode):
      encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

      for column in columns_to_encode:
          encoder.fit(self.x_train[[column]])

          train_encoded = encoder.transform(self.x_train[[column]])
          train_encoded_df = pd.DataFrame(
            train_encoded,
            columns=[f"{column}_{cat}" for cat in encoder.categories_[0]],
            index=self.x_train.index
          )

          test_encoded = encoder.transform(self.x_test[[column]])
          test_encoded_df = pd.DataFrame(
            test_encoded,
            columns=[f"{column}_{cat}" for cat in encoder.categories_[0]],
            index=self.x_test.index
          )

          self.x_train = pd.concat([self.x_train.drop(columns=[column]), train_encoded_df], axis=1)
          self.x_test = pd.concat([self.x_test.drop(columns=[column]), test_encoded_df], axis=1)
      
    def createModel(self):
         self.RF_class = RandomForestClassifier(n_estimators= 300, min_samples_split= 2, min_samples_leaf= 4, max_samples =None, max_features=None, max_depth= 20, criterion='gini', bootstrap= True)
    def train_model(self):
        self.RF_class.fit(self.x_train, self.y_train)
    def makePrediction(self):
        self.y_predict = self.RF_class.predict(self.x_test)              
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['Canceled','Not_Canceled']))

    def save_model_to_file(self, save_path, filename):
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'wb') as file:
            joblib.dump(self.RF_class, file, compress=3)
    

file_path = 'C:/Users/Jeremy/Downloads/MD/Dataset_B_hotel.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.drop_missing_data()
data_handler.drop_column('Booking_ID')
data_handler.label_encode('booking_status')
data_handler.label_encode('arrival_year')
data_handler.create_input_output('booking_status')

input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
columns_to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
model_handler.one_hot_encode(columns_to_encode)
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()
savepath = "C:/Users/Jeremy/Downloads/MD"
model_handler.save_model_to_file(savepath, 'RF_class.pkl')
training_columns = model_handler.x_train.columns.tolist()
with open(os.path.join(savepath, 'training_columns.pkl'), 'wb') as f:
    pickle.dump(training_columns, f)