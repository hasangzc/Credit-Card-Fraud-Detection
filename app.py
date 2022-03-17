# Import the modules
from argparse import ArgumentParser

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from model.logisticregression import LogisticRegressionProd
from train import declareParserArguments

# Create Flask App
app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        parser = ArgumentParser(
            description="Recognize Fraudulent Credit Card Transactions"
        )
        args = declareParserArguments(parser=parser)
        # Get the data from the POST request.
        data = request.get_json(force=True)
        data = pd.DataFrame(data)
        # Create LogisticRegressionProd object
        log_reg = LogisticRegressionProd(args=args)
        # Get predictions
        predictions = log_reg.test(df=data)
        # Add predictions as a column
        data["Predictions"] = predictions
        # Delete not needed values
        del predictions
        # Get predictions as a list in json format
        # return jsonify({"Predictions": list(predictions)})
        # Get dataframe with predictions(json)
        # return Response(data.to_json(orient="records"), mimetype="application/json")
        return Response(data.to_xml(), mimetype="text/xml")
    else:
        return "Error occured while processing your data into the model!", 400


if __name__ == "__main__":
    app.run(port=3000, debug=True)
