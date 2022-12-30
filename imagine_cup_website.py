import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle

#creating flask app
index = Flask(__name__)

#load pickle model
mod = pickle.load(open("imaginecup.pkl","rb"))

@index.route("/")
def Home():
    return render_template("index.html")

@index.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = mod.predict(features)

    return render_template("index.html",prediction_text = "The heart failure prediction is {}".format(prediction))

if __name__ == "__main__":
    index.run(debug=True)
