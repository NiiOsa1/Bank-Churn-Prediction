Bank Churn Prediction: A Neural Network Classifier

This project predicts customer churn for a bank using a Neural Network-based classifier. The goal is to identify at-risk customers and implement proactive retention strategies. Leveraging machine learning and detailed exploratory analysis, this project delivers actionable insights for customer retention.

Table of Contents

1. Project Overview
2. Objective
3. Features
4. Data Dictionary
5. Project Structure
6. Installation
7. Usage
8. Model Architecture and Training
9. Results and Business Insights
10. Contributions
11. License
12. Contact

Project Overview

* Problem: Businesses like banks face a critical challenge in retaining customers, commonly referred to as customer churn.

* Objective: Understand factors influencing customer churn and predict customers at risk of leaving the bank within the next six months.

* Solution: Implement a Neural Network Classifier to predict churn and derive business insights for targeted interventions.

Objective

* This project aims to build a Neural Network-based classifier to predict whether a customer will leave the bank. By identifying at-risk customers, banks can reduce churn, enhance customer satisfaction, and retain valuable clientele.

Features

1. Data Analysis and Visualization

* Exploratory Data Analysis (EDA) to uncover patterns and correlations.

* Visualizations of key trends using Matplotlib and Seaborn.

2. Preprocessing Pipeline

* Feature engineering: Label encoding, one-hot encoding, and standard scaling.

* Handling missing values and imbalanced datasets with SMOTE.

3. Neural Network Model

* Custom-built Neural Network using TensorFlow and Keras.

* Flexible architecture with varying layers and activation functions.

4. Evaluation Metrics

* Metrics: Accuracy, recall, precision, F1-score.

* Optimization with custom thresholds to balance recall and precision.

Data Dictionary
Feature	                         Description
CustomerId	              Unique identifier for each customer
Surname	                      Last name of the customer
CreditScore                   Credit history score
Geography	              Customer's location
Gender	                      Gender of the customer
Age	                      Age of the customer
Tenure	                      Number of years with the bank
NumOfProducts	              Number of products purchased
Balance	Account               balance
HasCrCard	              Whether the customer has a credit card
IsActiveMember	              Whether the customer is an active member
EstimatedSalary	              Estimated annual salary
Exited	                      Target variable (1 = Churn, 0 = Retained)
Project Structure


The project is structured following the cookiecutter data science template:

plaintext
Copy code
├── data
│   ├── raw              <- The original, immutable data dump.
│   ├── processed        <- The final, canonical datasets for modeling.
│
├── notebooks            <- Jupyter notebooks.
│   ├── Bank_Churn_Prediction.ipynb
│
├── src
│   ├── data             <- Scripts to process and transform data.
│   ├── models           <- Scripts to train and predict models.
│   ├── visualization    <- Scripts to generate plots and figures.
│
├── README.md            <- The top-level README for describing the project.


Installation

1. Clone the repository:

bash

git clone https://github.com/NiiOsa1/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction

2. Install the required libraries:

bash

pip install -r requirements.txt


Usage

1. Prepare the Dataset: Ensure the dataset is in the data/raw/ directory.

2. Run the Notebook: Open notebooks/Bank_Churn_Prediction.ipynb in Jupyter and execute the cells sequentially.

3. Visualize Results: Analyze churn trends and model evaluation metrics generated in the notebook.

4. Adjust Thresholds: Customize classification thresholds to balance recall and precision.


Model Architecture and Training

Architecture

* Input Layer: Number of features after preprocessing.

* Hidden Layer 1: 14 neurons, ReLU activation.

* Hidden Layer 2: 7 neurons, ReLU activation.

* Output Layer: 1 neuron, Sigmoid activation for binary classification.


Training Details

* Optimizer: Stochastic Gradient Descent (SGD).

* Loss Function: Binary Crossentropy.
 
* Metrics: Recall.


Performance

* Training F1 Score: ~0.73

* Validation F1 Score: ~0.71


Results and Business Insights

Key Insights:

* High Recall Performance:
Recall >90% ensures most churners are identified for proactive interventions.

* Feature Importance:
Geography, NumOfProducts, and IsActiveMember significantly influence churn.

Recommendations:

* Retention Campaigns:
Offer discounts or personalized services to at-risk customers.

* Localized Interventions:
Address churn trends by geography to tailor strategies.


Contributions
I welcome suggestions! If you'd like to enhance the project or add new features, feel free to copy the repository, make changes, and submit a pull request.

License
This project is licensed under the MIT License.

Contact
For questions or collaboration inquiries, please contact:

Name: Michael Mensah Ofeor
Email: michaelofeor2011@yahoo.com
GitHub: NiiOsa1

