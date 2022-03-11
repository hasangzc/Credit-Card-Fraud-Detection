from argparse import ArgumentParser
from os import remove

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy import stats


def DataPipeline(df: pd.DataFrame, args: ArgumentParser):
    # Check informations about data
    if args.data_informations:
        data_informations(df=df)
        print("Data informations ops finished!")
    # Seperate data
    df_fraud = df[df["Class"] == 1]
    df_normal = df[df["Class"] == 0]
    for column in df.columns:
        # The target variable is too unbalanced. For these reasons, the outlier operation was performed only for targets with a value of 0.
        df_normal = remove_outliers_using_quantiles(
            qu_dataset=df_normal, qu_field=column, qu_fence="outer"
        )
    print("Remove outliers ops finished!")
    df = pd.concat([df_fraud, df_normal])
    df.sort_values("Time")

    # Since most of data has already been scaled. Scale the columns that are left to scale (Amount and Time)
    df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
    df["Time"] = StandardScaler().fit_transform(df["Time"].values.reshape(-1, 1))
    print("Standart Scalar ops finished!")
    # Return the dataframe formed after the operations done
    return df


def remove_outliers_using_quantiles(qu_dataset, qu_field, qu_fence):
    """With this fuction, remove outliers by IQR Values.
    Args:
        qu_dataset (pd.DataFrame): The given dataframe.
        qu_field (str): Column name to remove outlier.
        qu_fence('inner' or 'outer'): With this parameter, the coefficient is determined to determine the cutoff. inner:1.5 outer:3
    Returns:
        This function returns dataframe with outliers removed.
    """
    a = qu_dataset[qu_field].describe()

    # Calculate ıqr value
    iqr = a["75%"] - a["25%"]
    # print("interquartile range:", iqr)

    # Determine fences 1.5 iqr
    upper_inner_fence = a["75%"] + 1.5 * iqr
    lower_inner_fence = a["25%"] - 1.5 * iqr
    # print("upper_inner_fence:", upper_inner_fence)
    # print("lower_inner_fence:", lower_inner_fence)

    # Detemine fences 5 iqr
    upper_outer_fence = a["75%"] + 3 * iqr
    lower_outer_fence = a["25%"] - 3 * iqr
    # print("upper_outer_fence:", upper_outer_fence)
    # print("lower_outer_fence:", lower_outer_fence)

    # Count otliers
    count_over_upper = len(qu_dataset[qu_dataset[qu_field] > upper_inner_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field] < lower_inner_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    # print("percentage of records out of inner fences: %.2f" % (percentage))

    count_over_upper = len(qu_dataset[qu_dataset[qu_field] > upper_outer_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field] < lower_outer_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    # print("percentage of records out of outer fences: %.2f" % (percentage))

    if qu_fence == "inner":
        output_dataset = qu_dataset[qu_dataset[qu_field] <= upper_inner_fence]
        output_dataset = output_dataset[output_dataset[qu_field] >= lower_inner_fence]
    elif qu_fence == "outer":
        output_dataset = qu_dataset[qu_dataset[qu_field] <= upper_outer_fence]
        output_dataset = output_dataset[output_dataset[qu_field] >= lower_outer_fence]
    else:
        output_dataset = qu_dataset

    # print("length of input dataframe:", len(qu_dataset))
    # print("length of new dataframe after outlier removal:", len(output_dataset))

    return output_dataset


def data_informations(df: pd.DataFrame):
    """With this function, information about the data can be obtained.
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
         It will print information about data.
    """
    print("                          DATA INFORMATIONS")
    print("\n\n")
    print(f"Shape of data: rows:{df.shape[0]} columns:{df.shape[1]}")
    print("\n\n")
    print(f"First 5 rows: {df.head()}")
    print("\n\n")
    print(f"Columns and dtypes: {df.info()}")
    print("\n\n")
    print(f"Properties like count, mean, std: {df.describe().T}")
    print("\n\n")
    print(f"Null values: {df.isnull().sum().sum()}")
    print("\n\n")
    print("Target Variable value counts;")
    print(
        f"No Frauds', {df['Class'].value_counts()[0]/len(df) * 100}, % of the dataset"
    )
    print(f"Frauds', {df['Class'].value_counts()[1]/len(df) * 100}, % of the dataset")
