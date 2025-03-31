
import streamlit as st

def hyperparameter_filters():
    st.sidebar.title("ANN Hyperparameters")
    params = {
        "num_layers": st.sidebar.slider("Hidden Layers", 1, 5, 2),
        "neurons_per_layer": st.sidebar.slider("Neurons per Layer", 8, 256, 64, step=8),
        "activation": st.sidebar.selectbox("Activation Function", ["relu", "tanh", "sigmoid"]),
        "dropout_rate": st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3),
        "batch_norm": st.sidebar.checkbox("Use Batch Normalization", value=True),
        "optimizer": st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"]),
        "learning_rate": st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001),
        "batch_size": st.sidebar.selectbox("Batch Size", [16, 32, 64]),
        "epochs": st.sidebar.slider("Epochs", 5, 100, 20)
    }
    return params
