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

from num2words import num2words

import weather

import scipy.sparse
import csv

from run import remove_names_and_date, merge_weather_isw_region

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    date = request.form['date']

    query = datetime.strptime(date, '%Y-%m-%d')

    time = request.form['time']
    time_query = datetime.strptime(time, "%H:%M")

    chosen_date = datetime.combine(query.date(),time_query.time())

    region = request.form['region']

    if region == 'All':
        region = 'Kyiv'

    city = region
    df_regions = pd.read_csv("data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")
    index = df_regions.index[df_regions["center_city_en"] == city].tolist()[0] if city in df_regions["center_city_en"].values else df_regions.index[df_regions["center_city_ua"] == city].tolist()[0]

    region_df = df_regions.iloc[index]

    text_df = utils.get_text_df(chosen_date - timedelta(days=1))
    text_df["date"] = chosen_date.date() - timedelta(days=1)

    weather_forecast_df = weather.vectorize(weather.get_weather_forecast(chosen_date.isoformat(), region_df.at["center_city_en"]+",UA"))

    weather_forecast_df["day_datetime"] = pd.to_datetime(weather_forecast_df["day_datetime"])
    weather_forecast_df["city"] = weather_forecast_df["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
    weather_forecast_df["city"] = weather_forecast_df["city"].replace("Хмельницька область", "Хмельницький")
    input_df = merge_weather_isw_region(weather_forecast_df, text_df, pd.DataFrame(region_df).transpose())

    with open("D:/Proga/model/Series model/best_RandomForestClassifier_V.pkl", "rb") as modelfile:
        clf = pickle.load(modelfile, encoding="utf-8")
        schedule = clf.predict(input_df)

    start_time = time_query

    time_array = []

    for i in range(12):
        time_str = datetime.strftime(start_time + timedelta(hours=i), '%H:00')
        time_array.append(time_str)

    return render_template('result.html', chosen_date = chosen_date, region = region, schedule = schedule, time_array = time_array)


if __name__ == '__main__':
    app.run(debug=True)
