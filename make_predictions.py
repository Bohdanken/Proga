import numpy as np
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
import os

from num2words import num2words

import weather

import scipy.sparse
import csv

from run import remove_names_and_date, merge_weather_isw_region

app = Flask(__name__)
time=datetime.now()
path_to_forecasts="./data/forecasts/"
time_as_path='D:/Proga/data/forecasts/2023-04-24 0.json'
@app.route('/new_prediction', methods=['GET'])
def calculate():
    time_now = datetime.now()
    cutoff_time = time_now + timedelta(hours=12)
    data = request.get_json()
    param=data['file_path_from_project_root']
    name=""
    if param=="":
        name = str(time.date()) + " " + str(time.hour) + "->" + str(cutoff_time.date()) + " " + str(cutoff_time.hour)
    else:
        name=param
    #if (time_now - time) < timedelta(hours=12):
        #get_forecast_for_all()
    weather_forecast_df_all_regions = pd.read_json(time_as_path)#time_as_path
    schedules = []
    df_regions = pd.read_csv(
        "data" + sep + "weather_alarms_regions" + sep + "regions.csv", sep=",")
    groups = weather_forecast_df_all_regions.groupby(weather_forecast_df_all_regions.index // 24)
    text_df = utils.get_text_df(time - timedelta(days=1))
    text_df["date"] = time.date() - timedelta(days=1)
    for index, forecast in groups:
        forecast["day_time_compare"] = pd.to_datetime(forecast['day_datetime'].str.split().str[0] + ' ' +forecast['hour_datetime'])
        forecast["day_datetime"] = pd.to_datetime( forecast["day_datetime"])
        filter1=forecast["day_time_compare"] <= cutoff_time
        filter2=forecast['day_time_compare'] >= time_now
        filter=filter1 & filter2
        forecast = forecast.drop("day_time_compare", axis=1)
        filtered_forecast = forecast[filter]
        filtered_forecast.reset_index(drop=True)
        filtered_forecast["city"] = filtered_forecast["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
        filtered_forecast["city"] = filtered_forecast["city"].replace("Хмельницька область", "Хмельницький")
        input_df = merge_weather_isw_region(filtered_forecast, text_df, df_regions)

        with open("./model/Series model/best_RandomForestClassifier_V.pkl", "rb") as modelfile:
            clf = pickle.load(modelfile, encoding="utf-8")
            schedule = clf.predict(input_df)
            schedules.extend(schedule.tolist())
        predictions = np.reshape(schedules, (25, 12))

        time_array = []
        for i in range(12):
            time_str = datetime.strftime( time_now+ timedelta(hours=i), '%H:00')
            time_array.append(time_str)
        # save predictions to file
        with open(f"data/predicted_alarms/{name}.pkl", "wb") as f:
            pickle.dump(predictions, f)
        return response


@app.route('/update', methods=['POST'])
def get_forecast_for_all():
    df_regions = pd.read_csv("data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")
    df_forecast=pd.DataFrame()
    chosen_date=datetime.now()
    for i, region in enumerate(df_regions["center_city_en"]):
        weather_forecast_df = weather.vectorize(weather.get_weather_forecast(chosen_date.isoformat(), region+",UA"))
        df_forecast=pd.concat([df_forecast,weather_forecast_df],axis=0)
    df_forecast=df_forecast.reset_index(drop=True)
    print(len(df_forecast["city_resolvedAddress"].unique()))
    global time
    time = chosen_date
    global time_as_path
    time_as_path =path_to_forecasts+str(str(chosen_date.date())+" "+str(chosen_date.hour)+".json",)
    df_forecast.to_json(path_or_buf=time_as_path)

if __name__ == '__main__':
    #get_forecast_for_all()
    app.run(debug=True)
