
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from Filter import hyperparameter_filters
from Connector import load_and_train_model
from ModelSummary import model_summary_to_df

st.set_page_config(page_title="Amazon Product ANN Dashboard", layout="wide")
st.title("Amazon Product Category Classifier 🛍️")

params = hyperparameter_filters()
model, history, X_test, y_test, y_pred_labels, label_names = load_and_train_model("Amazon_Products_Cleaned_INR.csv", "main_category", params)

# Display Model Summary
st.subheader("Model Summary")
summary_string = []
model.summary(print_fn=lambda x: summary_string.append(x))
st.code("\n".join(summary_string))

# Training History
st.subheader("Training Accuracy & Loss")
history_df = pd.DataFrame(history.history)
st.line_chart(history_df[["accuracy", "val_accuracy"]])
st.line_chart(history_df[["loss", "val_loss"]])

# Evaluation
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_labels)
fig, ax = plt.subplots(figsize=(10, 10))  # 🔧 Resize the figure
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
    cbar=True
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# 🔧 Rotate x-axis and y-axis labels for readability
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

st.pyplot(fig)

st.subheader("Classification Report")
from sklearn.utils.multiclass import unique_labels
valid_labels = unique_labels(y_test, y_pred_labels)

report = classification_report(
    y_test,
    y_pred_labels,
    labels=valid_labels,
    target_names=[label_names[i] for i in valid_labels],
    output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose())
