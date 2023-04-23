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
time=datetime.now()
path_to_forecasts="./data/forecasts/"
time_as_path=''
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

    with open("data/predictions/predictions.pkl", "rb") as f:
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

    if region == 'All':
       # if (datetime.now() - time) < timedelta(hours=12):
        #    get_forecast()
        weather_forecast_df_all_regions = pd.read_json("./data/forecasts/2023-04-23 5.json")#time_as_path)
        schedules = []
        df_regions = pd.read_csv(
            "data" + sep + "weather_alarms_regions" + sep + "regions.csv", sep=",")
        groups = weather_forecast_df_all_regions.groupby(weather_forecast_df_all_regions.index // 24)
        for index, forecast in groups:
            city =forecast["city_address"].unique()[0].split(",")[0]
            index = df_regions.index[df_regions["center_city_en"] == city].tolist()[0]
            region_df = df_regions.iloc[index]
            cutoff_time = datetime.now() + timedelta(hours=12)
            forecast["day_datetime"] = pd.to_datetime(forecast["day_datetime"])
            filter1=forecast["day_datetime"] <= cutoff_time
            filter2=forecast['day_datetime'] >= datetime.now()
            filter=filter1 & filter2
            filtered_forecast = forecast[filter]
            filtered_forecast.reset_index(drop=True)
            filtered_forecast["city"] = filtered_forecast["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
            filtered_forecast["city"] = filtered_forecast["city"].replace("Хмельницька область", "Хмельницький")

            text_df = utils.get_text_df(time - timedelta(days=1))
            text_df["date"] = time.date() - timedelta(days=1)

            input_df = merge_weather_isw_region(filtered_forecast, text_df, pd.DataFrame(region_df).transpose())

            with open("./model/Series model/best_RandomForestClassifier_V.pkl", "rb") as modelfile:
                clf = pickle.load(modelfile, encoding="utf-8")
                schedule = clf.predict(input_df)
                schedules.append(schedule.tolist())

        time_array = []
        regions = df_regions['center_city_en'].tolist()
        for i in range(12):
            time_str = datetime.strftime(time + timedelta(hours=i), '%H:00')
            time_array.append(time_str)
        # save predictions to file
        with open("data/predictions/predictions.pkl", "wb") as f:
            pickle.dump(schedules, f)

        with open("data/predictions/time.pkl", "wb") as f:
            pickle.dump(time, f)

        with open("data/predictions/time_array.pkl", "wb") as f:
            pickle.dump(time_array, f)

        with open("data/predictions/regions.pkl", "wb") as f:
            pickle.dump(regions, f)

        return render_template('result_all.html', chosen_date=chosen_date, regions=regions,
                               predictions=schedules, time_array=time_array, region_idx=0)
    else:
        city = region
        df_regions = pd.read_csv(
            "data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")
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

        start_time = time_query

        time_array = []

        for i in range(12):
            time_str = datetime.strftime(
                start_time + timedelta(hours=i), '%H:00')
            time_array.append(time_str)

        return render_template('result.html', chosen_date=chosen_date, region=region,
                               schedule=schedule, time_array=time_array)


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
    #get_forecast()
    app.run(debug=True)
