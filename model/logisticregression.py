# Import the modules
import warnings
from argparse import ArgumentParser
from pathlib import Path
from pickle import dump, load
from typing import NoReturn

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from preprocessing import DataPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")


class LogisticRegressionTrainer:
    """
    This class process the data, trains a Logistic Regression Model and test the trained model when necessary.
    """

    def __init__(self, args: ArgumentParser) -> NoReturn:
        """Init method.

        Args:
            args (ArgumentParser): The arguments of the training and testing session.

        Returns:
            NoReturn: This method does not return anything.
        """
        # Create the args object
        self._args = args

        # Process the data
        self._df = DataPipeline(pd.read_csv(f"./data/{args.data}.csv"), args=args)
        # Detach Target and other features
        self._X_features = self._df.loc[:, :"Amount"]
        self._y_target = self._df.iloc[:, -1]
        # Data is very unbalanced so I'm going to use a smote.
        self._smote = SMOTE(random_state=0)
        self._smote_x, self._smote_y = self._smote.fit_resample(
            self._X_features, self._y_target
        )
        # Test, Train split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._smote_x,
            self._smote_y,
            test_size=0.3,
            random_state=0,
            stratify=self._smote_y,
        )

    def _grid_search_then_train(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> LogisticRegression:
        """This function searches for the best estimator.

        Args:
            X_train (pd.DataFrame): The training data.
            y_train (pd.DataFrame): The testing data.

        Returns:
            LogisticRegression: The best estimator.
        """
        # parameter grid
        parameters = {
            "penalty": ["l1", "l2"],
            "C": np.logspace(-3, 3, 7),
            "solver": ["newton-cg", "lbfgs", "liblinear"],
        }
        # Declare the model
        logreg = LogisticRegression()
        # Declare the GridSearchCv object
        clf = GridSearchCV(logreg, param_grid=parameters, scoring="accuracy", cv=10)
        # Fit the object
        clf.fit(x_train, y_train)
        return clf.best_estimator_

    def _train(self) -> NoReturn:
        """This function train an LogisticRegression model.

        Returns:
            NoReturn: This function does not return anything.
        """
        # Declare a saving path
        self.model_path = f"./saved_models/LogisticRegModel/"
        # Create the saving path if does it not exist
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        # Search for best parameters
        self.bst = self._grid_search_then_train(
            x_train=self._X_train, y_train=self._y_train
        )
        # Save the model
        dump(
            obj=self.bst,
            file=open(f"{self.model_path}/logisticregressin_model.pkl", "wb"),
        )

    def _test(self) -> NoReturn:
        """This function tests the test data using the last trained model in train.py procedure.

        Returns:
            NoReturn: This function does not return anything.
        """
        # Predict the test data
        predictions = self.bst.predict(self._X_test)
        # Calculate metrics for pretiction values and actual values
        matrix = confusion_matrix(self._y_test, predictions)
        accuracy = accuracy_score(self._y_test, predictions)
        recall = recall_score(self._y_test, predictions)
        roc_auc = roc_auc_score(self._y_test, predictions)
        precision = precision_score(self._y_test, predictions)
        f1 = f1_score(self._y_test, predictions)
        # Print metrics results
        print(
            f"Test Data Results: confusion_matrix: {matrix}, accuracy_score:{accuracy}"
        )
        print("\n")
        print(f"recall_score:{recall}, roc_auc_score:{roc_auc}")
        print("\n")
        print(f"precision_score={precision}, f1_score:{f1}")

    def pipeline(self) -> NoReturn:
        # Train the model
        self._train()
        if self._args.is_testing:
            # Test the model
            self._test()


# After training check test results and plot them!
# Production
class LogisticRegressionProd:
    """
    This class loads a trained model and supplies prediction method for the production.
    """

    def __init__(self, args: ArgumentParser) -> NoReturn:
        """Init method.

        Args:
            args (ArgumentParser): The arguments of the production.

        Returns:
            NoReturn: This method does not return anything.
        """
        # Create the args object
        self._args = args
        # Declare a model path to load
        self.model_path = f"./saved_models/LogisticRegModel/logisticregressin_model.pkl"
        # Load the model
        self._bst = self._load_model()

    def test(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function tests the production data using the given trained model

        Returns:
            ndarray: The predicted values
        """
        # Process the dataframe
        df = DataPipeline(df=df, args=self._args)
        # If the predictions are the visualization
        if self._args.visualize_log_results:
            # Copy the data as it is to be returned
            df_vis = df.copy()
            return (self._bst.predict(X=df.loc[:, df.columns != "Class"]), df_vis)
        # Otherwise, return the prediction only
        # return self._bst.predict(self._bst.predict(X=df.loc[:, df.columns != "Class"]))
        return (self._bst.predict(X=df.loc[:, df.columns != "Class"])).astype(float)

    def _load_model(self) -> LogisticRegression:
        """This method loads an LogisticRegression model from the given path.

        Returns:
            LogisticRegression: The loaded version of Logistic Regression.
        """
        # Load the model, then return
        return load(open(f"{self.model_path}", "rb"))
