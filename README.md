# Credit-Card-Fraud-Detection
### About The Project
A model has been developed that finds anonymized credit card transactions that have been labeled as fake or real. In this project, various images were obtained by analyzing data and a prediction model was created with Logistic Regression. With pip install requirements_linux.txt , you can install the modules and libraries used with their versions.

### About Data
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

### Built with
* Python

### Some Visualizations Before Data Operation
<img src="visualization_results/About_Data/boxplots.png" width=800 height=400>
