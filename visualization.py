import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import NoReturn

import xlsxwriter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from plotly.offline import iplot
from termcolor import cprint

from preprocessing import DataPipeline
from train import declareParserArguments

# Helper links
# https://www.kaggle.com/kadirduran/fraud-detection-with-deployment
# https://www.kaggle.com/vincentlugat/lightgbm-plotly
# https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow/notebook
# https://www.kaggle.com/enkidoctordu/eda-and-machine-learning-prediction-of-credit-card


def target_distribution(df: pd.DataFrame):
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
    plt.savefig(
        f"./visualization_results/About_Data/target_distribution.png",
        bbox_inches="tight",
    )


def feature_distribution(df: pd.DataFrame, column_name: str):
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
    plt.savefig(
        f"./visualization_results/About_Data/{column_name}_distribution.png",
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
    null_feat = pd.DataFrame(
        len(df[column_name]) - df.isnull().sum(), columns=["Count"]
    )
    percentage_null = (
        pd.DataFrame(
            (len(df[column_name]) - (len(df[column_name]) - df.isnull().sum()))
            / len(df[column_name])
            * 100,
            columns=["Count"],
        )
    ).round(2)

    trace = go.Bar(
        x=null_feat.index,
        y=null_feat["Count"],
        opacity=0.8,
        text=percentage_null["Count"],
        textposition="auto",
        marker=dict(color="#7EC0EE", line=dict(color="#000000", width=1.5)),
    )

    layout = dict(title="Missing Values (count & %)")
    fig = go.Figure(dict(data=[trace], layout=layout))
    iplot(fig, filename="Missing Values")
    fig.write_html("./visualization_results/About_Data/missing_values.html")


def pca_columns_outliers(df: pd.DataFrame):
    plt.style.use("ggplot")
    f, ax = plt.subplots(figsize=(11, 15))

    ax.set_facecolor("#fafafa")
    ax.set(xlim=(-5, 5))
    plt.ylabel("Variables")
    plt.title("Outliers Pca columns")
    ax = sns.boxplot(
        data=df.drop(columns=["Amount", "Class", "Time"]), orient="h", palette="Set2"
    )
    plt.savefig(
        f"./visualization_results/About_Data/pca_columns_outliers.png",
        bbox_inches="tight",
    )


def box_plot_after_rmv_outliers(df: pd.DataFrame):
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
    # Save resulting plot
    plt.savefig(
        f"./visualization_results/About_Data/box_plots_after_ops.png",
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


def visualize_after_data_ops(df: pd.DataFrame, args=ArgumentParser) -> NoReturn:
    """With this function, data after operations is visualized with different methods.
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        NoReturn: This method does not return anything."""

    Path(f"./visualization_results/data_after_ops/").mkdir(parents=True, exist_ok=True)
    df.to_excel(
        f"./visualization_results/data_after_ops/data.xlsx",
        engine="xlsxwriter",
        index=False,
    )
    box_plot_after_rmv_outliers(df=df)


if __name__ == "__main__":
    parser = ArgumentParser(description="visualizations")
    args = declareParserArguments(parser=parser)

    # Ignore warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # py.init_notebook_mode(connected=True)
    # data before data ops
    df_before_ops = pd.read_csv(f"./data/{args.data}.csv")
    # data after data ops
    df_after_ops = DataPipeline(df=pd.read_csv(f"./data/{args.data}.csv"), args=args)
    # visualize
    visualize_before_data_ops(df=df_before_ops)
    visualize_after_data_ops(df=df_after_ops)
