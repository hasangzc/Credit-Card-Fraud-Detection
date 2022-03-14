# Import the modules
import numpy as np
from flask import Flask, request, jsonify, render_template

# Create Flask App
app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", method=["POST"])
def predict():
    pass
