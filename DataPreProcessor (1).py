
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import streamlit as st
import pandas as pd

@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


class DataPreProcessor:
    def __init__(self, dataframe, target_variable, use_one_hot_encoding=True, train_test_split_percentage=80):
        self.df = dataframe
        self.target = target_variable
        self.use_one_hot_encoding = use_one_hot_encoding
        self.train_size = train_test_split_percentage / 100.0

    def pre_process(self):
        y = self.df[self.target]
        X = self.df.drop(columns=[self.target])

        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Pipelines
        num_pipeline = Pipeline([('scaler', StandardScaler())])
        if self.use_one_hot_encoding:
            cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
        else:
            cat_pipeline = 'passthrough'

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        X_processed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, train_size=self.train_size, random_state=42)
        return X_train, X_test, y_train, y_test
