# Import the modules
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import xlsxwriter
from click import style
from plotly.offline import iplot
from sklearn.metrics import confusion_matrix
from termcolor import cprint

from model.logisticregression import LogisticRegressionProd, LogisticRegressionTrainer
from preprocessing import DataPipeline
from train import declareParserArguments


def target_distribution(df: pd.DataFrame, ops="Before_Smote"):
    # Make direction for result plot and dataframe
    Path(f"./visualization_results/About_Data/").mkdir(parents=True, exist_ok=True)
    # Create a figure
    plt.figure(figsize=(7, 5))
    # Count target variables
    sns.countplot(df["Class"])
    # Set title and labels
    plt.title("Class Count", fontsize=15)
    plt.xlabel("Is Fraud?", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    # Save the resulting plot
    if ops == "Before_Smote":
        plt.savefig(
            f"./visualization_results/About_Data/target_distribution.png",
            bbox_inches="tight",
        )
    if ops == "After_Smote":
        plt.savefig(
            f"./visualization_results/About_Data/target_distribution_after_smote.png",
            bbox_inches="tight",
        )


def feature_distribution(df: pd.DataFrame, column_name: str, ops="before"):
    # Create figure
    figsize = (15, 8)
    sns.set_style("ticks")
    # Create plot
    s = sns.FacetGrid(df, hue="Class", aspect=2.5, palette={0: "lime", 1: "black"})
    s.map(sns.kdeplot, column_name, shade=True, alpha=0.6)
    s.set(xlim=(df[column_name].min(), df[column_name].max()))
    s.add_legend()
    s.set_axis_labels(column_name, "proportion")
    s.fig.suptitle(column_name)
    # Save resulting plot
    if ops == "before":
        plt.savefig(
            f"./visualization_results/About_Data/{column_name}_distribution.png",
            bbox_inches="tight",
        )
    if ops == "after":
        plt.savefig(
            f"./visualization_results/About_Data/{column_name}_distribution_after_ops.png",
            bbox_inches="tight",
        )


def check_dtypes(df: pd.DataFrame):
    # Create a figure
    plt.figure(figsize=(6, 6))
    # Count data types in dataframe and plot as barplot
    ax = df.dtypes.value_counts().plot(
        kind="bar", grid=False, fontsize=10, color="grey"
    )
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/datatypes.png",
        bbox_inches="tight",
    )


def time_target_dist(df: pd.DataFrame):
    # Create subplots
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
    # Determine bins and create hist plots
    bins = 50
    ax1.hist(df.Time[df.Class == 1], bins=bins)
    ax1.set_title("Fraud")
    ax2.hist(df.Time[df.Class == 0], bins=bins)
    ax2.set_title("Normal")
    # Determine labels
    plt.xlabel("Time (in Seconda)")
    plt.ylabel("Number of Transactions")
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/time_target_distribution.png",
        bbox_inches="tight",
    )


def amount_target_dist(df: pd.DataFrame):
    # Create subplots
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
    # Determine bins and create hist plots
    bins = 30
    ax1.hist(df.Amount[df.Class == 1], bins=bins)
    ax1.set_title("Fraud")
    ax2.hist(df.Amount[df.Class == 0], bins=bins)
    ax2.set_title("Normal")
    # Set labels and scale for plots
    plt.xlabel("Amount ($)")
    plt.ylabel("Number of Transactions")
    plt.yscale("log")
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/amount_target_distribution.png",
        bbox_inches="tight",
    )


def correlation_matrix(df: pd.DataFrame):
    # Create a figure
    plt.figure(figsize=(40, 10))
    # Plot heatmap
    sns.heatmap(df.corr(), annot=True, cmap="tab20c")
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/correlation_matrix.png",
        bbox_inches="tight",
    )


def box_plots(df: pd.DataFrame):
    cprint("Boxplots", "green", "on_red", attrs=["bold"])
    index = 0
    plt.figure(figsize=(20, 20))
    for feature in df.columns[:30]:
        index += 1
        plt.subplot(8, 4, index)
        sns.boxplot(y=feature, x="Class", data=df, whis=3)
    # Prevent axis labels from overlapping
    plt.tight_layout()
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/boxplots.png",
        bbox_inches="tight",
    )


def missing_plot(df: pd.DataFrame, column_name: str):
    # Determine null values in the features
    null_feat = pd.DataFrame(
        len(df[column_name]) - df.isnull().sum(), columns=["Count"]
    )
    # Determine percantege of null
    percentage_null = (
        pd.DataFrame(
            (len(df[column_name]) - (len(df[column_name]) - df.isnull().sum()))
            / len(df[column_name])
            * 100,
            columns=["Count"],
        )
    ).round(2)

    # Plot it with Plotly lib
    trace = go.Bar(
        x=null_feat.index,
        y=null_feat["Count"],
        opacity=0.8,
        text=percentage_null["Count"],
        textposition="auto",
        marker=dict(color="#7EC0EE", line=dict(color="#000000", width=1.5)),
    )
    # Set plot properties
    layout = dict(title="Missing Values (count & %)")
    fig = go.Figure(dict(data=[trace], layout=layout))
    iplot(fig, filename="Missing Values")
    # Save resulting plot as html
    fig.write_html("./visualization_results/About_Data/missing_values.html")


def correlation_features_target(df: pd.DataFrame, ops="before"):
    # Create a figure
    plt.figure(figsize=(10, 4), dpi=100)
    # Calculate correlation between features and target
    corr = pd.DataFrame(df.corr().sort_values("Class", ascending=True)["Class"])
    # Plot it
    sns.barplot(x=corr.index, y=corr.values.flatten(), palette="Blues")
    # Set plot properties
    plt.title("Correleation between features and class")
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    # Prevent axis labels from overlapping
    plt.tight_layout()
    # Save resulting plot
    if ops == "before":
        plt.savefig(
            f"./visualization_results/About_Data/corr_between_target_and_features.png",
            bbox_inches="tight",
        )
    if ops == "after":
        plt.savefig(
            f"./visualization_results/About_Data/corr_between_target_and_features_after_ops.png",
            bbox_inches="tight",
        )


def pca_columns_outliers(df: pd.DataFrame):
    # Set plot style
    plt.style.use("ggplot")
    # Create subplots
    f, ax = plt.subplots(figsize=(11, 15))
    # Set plot properties
    ax.set_facecolor("#fafafa")
    ax.set(xlim=(-5, 5))
    plt.ylabel("Variables")
    plt.title("Outliers Pca columns")
    # Plot it
    ax = sns.boxplot(
        data=df.drop(columns=["Amount", "Class", "Time"]), orient="h", palette="Set2"
    )
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/pca_columns_outliers.png",
        bbox_inches="tight",
    )


def box_plot_after_rmv_outliers(df: pd.DataFrame, ops="before"):
    cprint("Boxplots", "green", "on_red", attrs=["bold"])
    index = 0
    plt.figure(figsize=(20, 20))
    for feature in df.columns[:30]:
        index += 1
        plt.subplot(8, 4, index)
        sns.boxplot(y=feature, x="Class", data=df, whis=3)
    # Prevent axis labels from overlapping
    plt.tight_layout()
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/box_plots_after_ops.png",
        bbox_inches="tight",
    )


def plot_confussion_matrix(df: pd.DataFrame, actual_column, predict_column):
    # Calculate confussion matrix for predictions and actual values
    matrix = confusion_matrix(df[actual_column], df[predict_column])
    # Create a figure
    plt.figure(figsize=(8, 6))
    # Plot confussion matrix
    ax = sns.heatmap(matrix, cmap="YlGnBu", annot=True, fmt="d")
    ax.set_title("Confusion Matrix")
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/Prediction_Results/confussion_matrix.png",
        bbox_inches="tight",
    )


def plot_feature_importance():
    # Create an LogisticRegressionProd object
    log_reg_prod = LogisticRegressionProd(args=args)
    # Get saved model
    model = log_reg_prod._bst
    # Get columns orginal dataframe
    df_original = pd.read_csv(f"./data/{args.data}.csv")
    df_original = df_original.loc[:, df_original.columns != "Class"]
    # Detect coefficient
    importance = model.coef_[0]
    # Add columns and importance in dataframe
    feature_names = df_original.columns
    feature_importance = pd.DataFrame(feature_names, columns=["feature"])
    feature_importance["importance"] = importance
    # Sort values by importance column
    feature_importance = feature_importance.sort_values(
        by=["importance"], ascending=False
    )
    # Plot
    ax = feature_importance.plot.barh(x="feature", y="importance")
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/Prediction_Results/feature_importance.png",
        bbox_inches="tight",
    )


def visualize_before_data_ops(df: pd.DataFrame, args=ArgumentParser) -> NoReturn:
    """With this function, data before operations is visualized with different methods.
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        NoReturn: This method does not return anything."""

    # Plot target distribution
    target_distribution(df=df)
    # Plot feature distribution
    feature_distribution(df=df, column_name="Time")
    # Check datatypes in dataframe
    check_dtypes(df=df)
    # Plot time-target distribution
    time_target_dist(df=df)
    # Plot amount-target distribution
    amount_target_dist(df=df)
    # Plot correlation matrix
    correlation_matrix(df=df)
    # Plot box plots for all features
    box_plots(df=df)
    # Plot and check missing values
    missing_plot(df=df, column_name="Class")
    # Plot pca column and examine outliers
    pca_columns_outliers(df=df)
    # Plot correlations between features and taget(class feature)
    correlation_features_target(df=df)


def visualize_after_pre_data_ops(df: pd.DataFrame, args=ArgumentParser) -> NoReturn:
    """With this function, data after operations is visualized with different methods.
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        NoReturn: This method does not return anything."""

    # Create a folder for dataframe after the ops
    Path(f"./visualization_results/data_after_ops/").mkdir(parents=True, exist_ok=True)
    # Save dataframe in Excel format
    df.to_excel(
        f"./visualization_results/data_after_ops/data.xlsx",
        engine="xlsxwriter",
        index=False,
    )
    # Plot box plots after pipeline ops
    box_plot_after_rmv_outliers(df=df)
    # Plot correlations between features and taget(class feature)
    correlation_features_target(df=df, ops="after")
    # Feature distribution after ops
    feature_distribution(df=df, column_name="Amount", ops="after")


def visualize_test_results(df: pd.DataFrame, args=ArgumentParser) -> NoReturn:
    """With this function, visualize test results and data using the trained model
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        NoReturn: This method does not return anything.
    """
    # Make direction for result plot and dataframe
    Path(f"./visualization_results/Prediction_Results/").mkdir(
        parents=True, exist_ok=True
    )

    # Save dataframe in Excel format
    df.to_excel(
        f"./visualization_results/Prediction_Results/predictions.xlsx",
        engine="xlsxwriter",
        index=False,
    )

    # Plot confussion matrix
    plot_confussion_matrix(
        df=df,
        actual_column="Class",
        predict_column="Predictions",
    )

    # Plot feature importance
    plot_feature_importance()


if __name__ == "__main__":
    parser = ArgumentParser(description="visualizations")
    args = declareParserArguments(parser=parser)

    # Ignore warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # py.init_notebook_mode(connected=True)
    # data before data ops
    df_before_ops = pd.read_csv(f"./data/{args.data}.csv")
    # data after pre data ops
    df_after_ops = DataPipeline(df=pd.read_csv(f"./data/{args.data}.csv"), args=args)

    # Create an LogisticRegressionTrainer
    logistic_reg_trainer = LogisticRegressionTrainer(args=args)
    # Add features df after smote ops
    df_smote = logistic_reg_trainer._smote_x
    # Add target values as a column after smote ops
    df_smote["Class"] = logistic_reg_trainer._smote_y
    # Visualize target distribution after smote ops
    target_distribution(df=df_smote, ops="After_Smote")

    # Visualize unmodified dataset
    visualize_before_data_ops(df=df_before_ops)

    # Visualize data undergoing predata changes
    visualize_after_pre_data_ops(df=df_after_ops)

    # args.visualize_log_results = True
    # Create an LogisticRegressionProd object
    log_reg_prod = LogisticRegressionProd(args=args)
    # Get predictions and dataframe
    predictions, df_test = log_reg_prod.test(df=df_before_ops)
    # Add Predict results as a column
    df_test["Predictions"] = predictions

    # Delete not needed values
    del predictions
    # Visualize test results based on Logistic Regression model
    visualize_test_results(df=df_test, args=args)
