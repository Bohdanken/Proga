from flask import Flask, render_template, request

from os import sep
import pickle
import pandas as pd
from datetime import date, datetime, timedelta
import utils
import requests
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np

from num2words import num2words

import weather

import scipy.sparse
import csv
import json

import os

from run import merge_weather_isw_region

API_TOKEN = ""

df_regions = pd.read_csv(
        "data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/<chosen_region>')
def alarm_forecast(chosen_region):

    with open("data/predictions/regions.pkl", "rb") as f:
        regions = pickle.load(f)

    with open("data/predictions/chosen_date.pkl", "rb") as f:
        chosen_date = pickle.load(f)

    chosen_date_string = chosen_date.isoformat()
    chosen_date_string = chosen_date_string.replace("-", "_").replace(":", "_")
    chosen_date_string = chosen_date_string[:-9] + "_" + chosen_date_string[-8:]

    with open(f"data/predictions/prediction_All_regions_{chosen_date_string}.pkl", "rb") as f:
        predictions = pickle.load(f)

    with open("data/predictions/time_array.pkl", "rb") as f:
        time_array = pickle.load(f)

    region_idx = regions.index(chosen_region)

    return render_template('result_all.html', chosen_date=chosen_date, regions=regions,
                           predictions=predictions, time_array=time_array, region_idx=region_idx)


@app.route('/calculate', methods=['POST'])
def calculate():
    date = request.form['date']
    query = datetime.strptime(date, '%Y-%m-%d')
    time1 = request.form['time']
    time_query = datetime.strptime(time1, "%H:%M")
    chosen_date = datetime.combine(query.date(), time_query.time())
    region = request.form['region']
    payload = {
    "region": region,
    "date" : date,
    "time" : time1
}
    headers = {'Content-type': 'application/json'}

    # Send the JSON payload to the API using a POST request
    response = requests.post('http://34.227.228.224:7000/send_prediction', data=json.dumps(payload), headers=headers)
    time_array=None
    predictions=None
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        response_json = response.json()
        predictions=response_json["regions_forecast"]
        time_array=[]
        for time in predictions:
            time_array.append(time)
    else:
        print(f"Request failed with status code {response.status_code}")

    return render_template('result.html', chosen_date=chosen_date, region=region,
                       schedule=predictions, time_array=time_array)

if __name__ == '__main__':
    # date = "2023-04-26"

    # query = datetime.strptime(date, '%Y-%m-%d')

    # time1 = "00:01"
    # time_query = datetime.strptime(time1, "%H:%M")

    # chosen_date = datetime.combine(query.date(), time_query.time())
    # get_predictions(chosen_date)
    app.run(debug=True)
