
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_ann(input_shape, num_classes, num_layers, neurons_per_layer,
              activation, dropout_rate, batch_norm, optimizer, learning_rate, **kwargs):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation, input_shape=(input_shape,)))
    if batch_norm: model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation))
        if batch_norm: model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation="softmax"))
    
    optimizers = {
        "adam": Adam(learning_rate),
        "sgd": SGD(learning_rate),
        "rmsprop": RMSprop(learning_rate)
    }
    model.compile(optimizer=optimizers[optimizer], loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, epochs):
    return model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=batch_size, epochs=epochs, verbose=1)
