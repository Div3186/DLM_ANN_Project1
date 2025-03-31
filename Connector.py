
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DataPreProcessor import DataPreProcessor
from model import build_ann, train_model

def custom_preprocess(dataframe, target_variable):
    df = pd.read_csv(dataframe)
    le = LabelEncoder()
    df = df.dropna(subset=[target_variable])
    df[target_variable] = le.fit_transform(df[target_variable].astype(str))
    preprocessor = DataPreprocessor(df, target_variable, use_one_hot_encoding=True)
    X_train, X_test, y_train, y_test = preprocessor.pre_process()
    return X_train, X_test, y_train, y_test, le

def load_and_train_model(DATASET_FILE, target_variable, hyperparams):
    X_train, X_test, y_train, y_test, label_encoder = custom_preprocess(DATASET_FILE, target_variable)
    num_classes = len(label_encoder.classes_)
    model = build_ann(
        input_shape=X_train.shape[1],
        num_classes=num_classes,
        **hyperparams
    )
    history = train_model(model, X_train, y_train, X_test, y_test,
                          batch_size=hyperparams.get("batch_size", 32),
                          epochs=hyperparams.get("epochs", 20))
    y_pred = model.predict(X_test).argmax(axis=1)
    return model, history, X_test, y_test, y_pred, label_encoder.classes_
