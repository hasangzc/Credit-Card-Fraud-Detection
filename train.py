# Import the modules
from argparse import ArgumentParser

from model.logisticregression import LogisticRegressionTrainer


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

    parser.add_argument(
        "--is_testing",
        action="store_true",
        default=False,
        help="Whether to test the trained model",
    )

    parser.add_argument(
        "--visualize_log_results",
        action="store_true",
        default=False,
        help="Whether to visualize the test results",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fraud Detection")
    # Add arguments
    args = declareParserArguments(parser=parser)
    # Test data and fetch metric results
    args.is_testing = True
    # Create an LogisticRegressionTrainer Object
    logistic_reg_trainer = LogisticRegressionTrainer(args=args)
    # Train LogisticRegression model
    logistic_reg_trainer.pipeline()
