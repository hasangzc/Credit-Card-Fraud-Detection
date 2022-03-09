from argparse import ArgumentParser

import pandas as pd

from preprocessing import DataPipeline


def declareParserArguments(parser: ArgumentParser) -> ArgumentParser:
    # Add arguments
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard",
        help="The data file name [without .csv]",
    )

    parser.add_argument(
        "--data_informations",
        action="store_true",
        default=False,
        help="Information about dataset.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fraud Detection")
    # Add arguments
    args = declareParserArguments(parser=parser)
    DataPipeline(pd.read_csv(f"./data/{args.data}.csv"), args=args)
