# Credit-Card-Fraud-Detection
### About The Project
A model has been developed that finds anonymized credit card transactions that have been labeled as fake or real. In this project, various images were obtained by analyzing data and a prediction model was created with Logistic Regression. With pip install requirements_linux.txt , you can install the modules and libraries used with their versions.

### About Data
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

### Built with
* Python

### Some Visualizations

* As seen from the boxplots, there are many outliers in the data;

<img src="visualization_results/About_Data/boxplots.png" width=600 height=300>


* IQR calculated and outliers removed from data. Boxplots after IQR operations;

<img src="visualization_results/About_Data/box_plots_after_ops.png" width=600 height=300>


* According to the target variable, you can examine the distribution plots of the features from the plot results.


* If we look at the correlation between the target and the features, we can see that the V17, V14, V12 and V11 features are highly correlated with the target value.

<img src="visualization_results/About_Data/corr_between_target_and_features.png" width=600 height=300>


* Dataset too unbalanced;

<img src="visualization_results/About_Data/target_distribution.png" width=600 height=300>


* The Smote operation was applied before training the data. Target distribution after Smote;

<img src="visualization_results/About_Data/target_distribution_after_smote.png" width=600 height=300>

* More detailed plots can be viewed in the visualization_results folder.
