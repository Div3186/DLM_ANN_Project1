# ANN Dashboard

Use the following link to access streamlit dashboard --> https://dlmannproject1-dpyns7f2nk5btswd4ewuxe.streamlit.app/

Use the buttons in filter pane for hyperparameter tuning

In case app takes too much time to run view this pdf of the dashboard --> https://github.com/user-attachments/files/19540834/055010_Amazon.Product.ANN.Dashboard.pdf

## 📘 Project Information

**Title** – Amazon Product Category Classifier using ANN

**Submitted by** – Divyank Harjani_055010 and Priyadarshani Dash_055033
## 📊 Description of Data

The dataset used in this analysis, *Amazon_Products_Cleaned_INR.csv*, contains information about various consumer products listed on Amazon India.

It includes the following attributes:

- **Product Title**
- **Selling Price**
- **MRP**
- **Brand**
- **Discount**
- **Product Category** (Target variable)

The target variable comprises **44 unique product categories**, including *Electronics, Handbags, Home Decor, Clothing, Toys*, and more.
## 🎯 Objective

The primary objective is to develop an **Artificial Neural Network (ANN)** model to predict the **product category** based on features such as brand, pricing, and discount.

This classifier can be used by e-commerce platforms to:
- Auto-categorize products uploaded by sellers
- Streamline inventory tagging
- Enable better filtering for customers
## ⚙️ Model Design & Hyperparameter Tuning

The model was built using the following customizable hyperparameters:
- Hidden Layers: 1 to 5
- Neurons per Layer: 8 to 256
- Activation Function: *ReLU, Tanh*
- Dropout Rate: 0.0 to 0.5
- Optimizer: *Adam, RMSprop*
- Learning Rate: 0.0001 to 0.01
- Batch Size: 16, 32, 64

**Preprocessing**:
- OneHotEncoding for categorical columns
- StandardScaler for numerical columns
- Train-test split ratio: 80:20
## 📈 Dashboard Overview

An interactive **Streamlit dashboard** was deployed using **Ngrok** (for temporary links) and **Streamlit Cloud** (for permanent deployment). Users can:
- Choose hyperparameters dynamically
- View real-time training history
- Inspect confusion matrix and classification report
- Evaluate model summary
## 🔍 Findings

- The model performed best with:
  - **2 hidden layers**
  - **64 neurons per layer**
  - **ReLU activation**
  - **Dropout: 0.3**
  - **Adam optimizer**
  - **Batch normalization enabled**

- The confusion matrix showed excellent performance for classes like *Handbags*, *Electronics*, and *Toys*, while struggling with less frequent categories like *Jams & Honey* or *Sex & Sensuality* due to class imbalance.

- Validation accuracy was observed around **85–90%**, suggesting strong generalization.
## 📌 Conclusion

This project demonstrates the power of ANN in text-based multi-class classification problems. The dynamic dashboard enables users to experiment with the model and observe real-time performance, making it a practical tool for **retail automation** and **e-commerce tagging systems**.
