# Credit Card Fraud Detection
## Problem Statement
Credit card fraud is a major challenge for financial institutions, with fraudulent transactions often hidden among millions of legitimate ones. This project applies Machine Learning and Anomaly Detection techniques to identify fraudulent transactions using the highly imbalanced Kaggle Credit Card Fraud Detection dataset.

The goal is to evaluate different supervised and unsupervised models, compare them using fraud-relevant metrics (Recall and F1-score), and identify the best-performing approach for detecting fraud.
## Dataset Details
The dataset is used from Kaggle : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and more background information about the data cannot be provided. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
## Libraries Used
| Library          | Purpose                                                            |
| ---------------- | ------------------------------------------------------------------ |
| **numpy**        | Efficient numerical computations and array operations              |
| **pandas**       | Data manipulation, cleaning, and analysis with DataFrames          |
| **matplotlib**   | Basic data visualization and plotting library                      |
| **seaborn**      | Advanced statistical data visualization built on matplotlib        |
| **scikit-learn** | Machine learning toolkit for preprocessing, modeling, evaluation   |
| **xgboost**      | Gradient boosting framework optimized for performance and accuracy |

## Methodology Used
1. Importing all required python libraries
2. Understanding the data
   -> Loaded the Kaggle dataset and explored its structure
   -> Studied the columns and the basic statistics
3. Data Pre-processing
   -> Checked for null values
   -> Checked for duplicated valued
4. EDA
   -> Scaled 'Time' and 'Amount' columns
   -> Visualized distribution of fraud vs non-fraud classes.
   -> Compared transaction amount/time patterns for fraud vs legitimate transactions.
   -> Visualised the correlation matrix for the entire dataset, legitimate transactions and fraud transactions
5. Created a balanced dataset
   -> Made a deep-copy of the origianl dataset
   -> Considered all 473 fraud transactions and 1000 legitimate transactions
6. Model development for both original and balanced dataset
   -> Dataset split for both original and balanced dataset
   -> Trained the models individually for both the datasets
       -> Logistic Regression
       -> Decision Tree
       -> Random Forest
       -> XgBoost
       -> Isolated Forest
       -> Local Outlier Factor
7. Model Evaluation
   Used multiple metrics for evaluation:
   -> Accuracy (overall correctness — but misleading due to imbalance).
   -> Recall (how many frauds were correctly identified — most important metric).
   -> F1-score (balance between precision and recall).
   -> Confusion Matrix for false positives/negatives.
   Here, in fraud detection scenerio, Recall and F1-score is prioritised instead of Accuracy, since missing fraud is costlier than flagging a normal transaction
## Results
-> Accuracy is misleading in fraud detection : Since the dataset is highly imbalanced, a model predicting all transactions as legitimate would still achieve ~99.8% accuracy, but completely fail to detect fraud.

-> Recall and F1-score are the key metrics:

1. Recall ensures that fraudulent transactions are correctly identified (minimizing false negatives).
2. F1-score balances precision and recall, ensuring the model not only catches fraud but also reduces false alarms.
3. High recall alone is not enough → If recall is high but F1 is low, the model may catch frauds but at the cost of too many false positives, making it impractical.
4. High F1 alone is not enough either → It could mean frauds are missed (low recall) even if predictions look balanced overall.
  
Therefore, combined evaluation of Recall + F1-score gives a more reliable measure of fraud detection performance than accuracy or individual metrics.
## Future Work
1. Implement deep learning (Autoencoders, LSTM) for anomaly detection.
2. Apply SMOTE / ADASYN for synthetic minority oversampling.
