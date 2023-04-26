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

def next_24_hours(time_query):
    start_time = time_query

    time_array = []

    for i in range(24):
        time_str = datetime.strftime(
            start_time + timedelta(hours=i), '%H:00')
        time_array.append(time_str)

    return time_array

def get_prediction(chosen_date, region):
    city = region
    index = df_regions.index[df_regions["center_city_en"] == city].tolist(
    )[0] if city in df_regions["center_city_en"].values else df_regions.index[df_regions["center_city_ua"] == city].tolist()[0]

    region_df = df_regions.iloc[index]

    text_df = utils.get_text_df(chosen_date - timedelta(days=1))
    text_df["date"] = chosen_date.date() - timedelta(days=1)

    weather_forecast_df = weather.vectorize(weather.get_weather_forecast(
        chosen_date.isoformat(), region_df.at["center_city_en"]+",UA"))

    weather_forecast_df["day_datetime"] = pd.to_datetime(
        weather_forecast_df["day_datetime"])
    weather_forecast_df["city"] = weather_forecast_df["city_resolvedAddress"].apply(
        lambda x: x.split(",")[0])
    weather_forecast_df["city"] = weather_forecast_df["city"].replace(
        "Хмельницька область", "Хмельницький")
    input_df = merge_weather_isw_region(
        weather_forecast_df, text_df, pd.DataFrame(region_df).transpose())

    with open("model"+sep+"8_random_forest_v2.pkl", "rb") as modelfile:
        clf = pickle.load(modelfile, encoding="utf-8")
        schedule = clf.predict(input_df)

    chosen_date_string = chosen_date.isoformat()
    chosen_date_string = chosen_date_string.replace("-", "_").replace(":", "_")
    chosen_date_string = chosen_date_string[:-9] + "_" + chosen_date_string[-8:]

    filename = f"data/predictions/prediction_{str(region)}_{chosen_date_string}.pkl"

    with open(filename, 'wb') as handle:
        pickle.dump(schedule, handle)
    return schedule

def get_predictions(chosen_date):
    chosen_date_string = chosen_date.isoformat()
    chosen_date_string = chosen_date_string.replace("-", "_").replace(":", "_")
    chosen_date_string = chosen_date_string[:-9] + "_" + chosen_date_string[-8:]

    text_df = utils.get_text_df(chosen_date - timedelta(days=1))
    text_df["date"] = chosen_date.date() - timedelta(days=1)

    df_forecast = pd.DataFrame()
    for i, region in enumerate(df_regions["center_city_en"]):
        weather_forecast_df = weather.vectorize(
            weather.get_weather_forecast(chosen_date.isoformat(), region+",UA"))
        df_forecast = pd.concat([df_forecast, weather_forecast_df], axis=0)
    df_forecast = df_forecast.reset_index(drop=True)

    df_forecast["day_datetime"] = pd.to_datetime(
        df_forecast["day_datetime"])
    df_forecast["city"] = df_forecast["city_resolvedAddress"].apply(
        lambda x: x.split(",")[0])
    df_forecast["city"] = df_forecast["city"].replace(
        "Хмельницька область", "Хмельницький")
    input_df = merge_weather_isw_region(df_forecast, text_df, df_regions)

    with open("model"+sep+"8_random_forest_v2.pkl", "rb") as modelfile:
        clf = pickle.load(modelfile, encoding="utf-8")
        schedule = clf.predict(input_df)

    predictions = np.reshape(schedule, (25, 24))

    # save predictions to file
    filename = f"data/predictions/prediction_All_regions_{chosen_date_string}.pkl"

    with open(filename, 'wb') as handle:
        pickle.dump(predictions, handle)
    return predictions



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

@app.route('/generate/get_json', methods=['POST'])
def calculate_endpoint():
    json_data = request.get_json()

    if json_data.get("token") is None:
        return {"error": "token is required"}, 400

    token = json_data.get("token")

    if token != API_TOKEN:
        return {"error": "wrong API token"}, 403

    date_string = ""
    if json_data.get("date"):
        date_string = json_data.get("date")

    try:
        query = datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        return {"error": "Please enter a valid date YY-mm-dd"}, 400

    time = ""
    if json_data.get("time"):
        time = json_data.get("time")
    try:
        time_query = datetime.strptime(time, "%H:%M")
    except ValueError:
        return {"error": "Please enter a valid time in the format HH:MM."}, 400

    today = datetime.combine(date.today() + timedelta(days=1),datetime.min.time())

    if (query < datetime(2023, 1, 20) or query > today):
        return {"error": "Please enter a valid date (between 2023-01-20 and tomorrow)"}, 400

    region = ""
    if json_data.get("region"):
        region = json_data.get("region")

    regions = df_regions['center_city_en'].tolist()

    if (region not in regions and region != "All"):
        return {"error": "Please enter a valid region"}, 400

    chosen_date = datetime.combine(query.date(), time_query.time())

    chosen_date_string = chosen_date.isoformat()
    chosen_date_string = chosen_date_string.replace("-", "_").replace(":", "_")
    chosen_date_string = chosen_date_string[:-9] + "_" + chosen_date_string[-8:]

    time_array = next_24_hours(time_query)

    if region == 'All':
        predictions = get_predictions(chosen_date)
        predictions = predictions.tolist()

        dictionary = {region: {time: prediction[i] for i, time in enumerate(time_array)} for region, prediction in zip(regions, predictions)}
        result = {
        "last_model_train_time": "2023-04-20T21:09:44Z",
        "model": "8_random_forest_v2",
        "date": chosen_date_string,
        "predictions": dictionary
        }
    else:
        predictions = get_prediction(chosen_date, region)
        predictions = predictions.tolist()

        dictionary = {}
        for time, prediction in zip(time_array, predictions):
            dictionary[time] = prediction


        result = {
        "last_model_train_time": "2023-04-20T21:09:44Z",
        "model": "8_random_forest_v2",
        "date": chosen_date_string,
        "region": region,
        "predictions": dictionary,
        }

    return result



@app.route('/calculate', methods=['POST'])
def calculate():
    date = request.form['date']

    query = datetime.strptime(date, '%Y-%m-%d')

    time1 = request.form['time']
    time_query = datetime.strptime(time1, "%H:%M")

    chosen_date = datetime.combine(query.date(), time_query.time())

    region = request.form['region']

    chosen_date_string = chosen_date.isoformat()
    chosen_date_string = chosen_date_string.replace("-", "_").replace(":", "_")
    chosen_date_string = chosen_date_string[:-9] + "_" + chosen_date_string[-8:]

    # df_regions = pd.read_csv(
    #     "data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")
    regions = df_regions['center_city_en'].tolist()

    time_array = next_24_hours(time_query)

    with open("data/predictions/chosen_date.pkl", "wb") as f:
        pickle.dump(chosen_date, f)

    with open("data/predictions/time_array.pkl", "wb") as f:
        pickle.dump(time_array, f)

    with open("data/predictions/regions.pkl", "wb") as f:
        pickle.dump(regions, f)

    if region == 'All':
        filename = f'prediction_All_regions_{chosen_date_string}.pkl'
        folder_path = 'data/predictions/'

        if filename in os.listdir(folder_path):
            with open(f"{folder_path}{filename}", "rb") as f:
                predictions = pickle.load(f)

            return render_template('result_all.html', chosen_date=chosen_date, regions=regions,
                               predictions=predictions, time_array=time_array, region_idx=0)
        else:
            predictions = get_predictions(chosen_date)

            return render_template('result_all.html', chosen_date=chosen_date, regions=regions,
                                predictions=predictions, time_array=time_array, region_idx=0)
    else:
        filename = f'prediction_{region}_{chosen_date_string}.pkl'
        folder_path = 'data/predictions/'

        if filename in os.listdir(folder_path):
            with open(f"{folder_path}{filename}", "rb") as f:
                schedule = pickle.load(f)

            return render_template('result.html', chosen_date=chosen_date, region=region,
                        schedule=schedule, time_array=time_array)
        else:
            schedule = get_prediction(chosen_date, region)

            return render_template('result.html', chosen_date=chosen_date, region=region,
                                schedule=schedule, time_array=time_array)


if __name__ == '__main__':
    # date = "2023-04-26"

    # query = datetime.strptime(date, '%Y-%m-%d')

    # time1 = "00:01"
    # time_query = datetime.strptime(time1, "%H:%M")

    # chosen_date = datetime.combine(query.date(), time_query.time())
    # get_predictions(chosen_date)
    app.run(debug=True)
