from flask import request, render_template
import flask
import numpy as np
import pickle
import pandas as pd
import xgboost

# Use pickle to load in the pre-trained model
with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__)

#cols = ['temperature', 'humidity', 'windspeed']

@app.route('/',methods=['GET', 'POST'])
def main():
     if flask.request.method == 'GET':
         # Just render the initial form, to get input
         return(flask.render_template('index.html'))
     if flask.request.method == 'POST':
        # Extract the input
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float)

        # Get the model's prediction
        prediction = int(model.predict(input_variables)[0])
        return render_template('index.html',pred='Expected no of bike to be rented:  {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
