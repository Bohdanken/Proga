from flask import Flask, render_template, request

from os import sep
import pickle
import pandas as pd
from datetime import date, datetime, timedelta
import utils


import weather


from run import remove_names_and_date, merge_weather_isw_region

app = Flask(__name__)

time=datetime.now()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/alarms', methods=['POST'])
def calculate():
    if (datetime.now()-time)<timedelta(hours=12):
        get_forecast()
    weather_forecast_df_all_regions=pd.read_json(str(time))
    schedules={}
    for index,forecast in weather_forecast_df_all_regions.iterrows():
        cutoff_time = datetime.now() - timedelta(hours=12)
        forecast["day_datetime"] = pd.to_datetime(forecast["day_datetime"])
        filtered_forecast = forecast[forecast['time'] >= cutoff_time]
        filtered_forecast.reset_index(drop=True)
        filtered_forecast["city"] = filtered_forecast["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
        filtered_forecast["city"] = filtered_forecast["city"].replace("Хмельницька область", "Хмельницький")

        text_df = utils.get_text_df(time - timedelta(days=1))
        text_df["date"] = time.date() - timedelta(days=1)

        input_df = merge_weather_isw_region(filtered_forecast, text_df, pd.DataFrame(filtered_forecast[""]).transpose())

        with open("D:/Proga/model/Series model/best_RandomForestClassifier_V.pkl", "rb") as modelfile:
            clf = pickle.load(modelfile, encoding="utf-8")
            schedule = clf.predict(input_df)
            schedules[filtered_forecast["city"]]=schedule

    time_array = []

    for i in range(12):
        time_str = datetime.strftime(time + timedelta(hours=i), '%H:00')
        time_array.append(time_str)

    return render_template('result.html', chosen_date = time, region = "All", schedules = schedules, time_array = time_array)


@app.route('/update', methods=['POST'])
def get_forecast():
    df_regions = pd.read_csv("data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")
    df_forecast=pd.DataFrame()
    chosen_date=datetime.now()
    for i, region in enumerate(df_regions["center_city_en"]):
        weather_forecast_df = weather.vectorize(weather.get_weather_forecast(chosen_date.isoformat(), region+",UA"))
        df_forecast=pd.concat([df_forecast,weather_forecast_df],axis=0)
    df_forecast=df_forecast.reset_index(drop=True)
    df_forecast.to_json(path_or_buf=str("Forecast"+".json"))
    global time
    time=chosen_date



if __name__ == '__main__':
    get_forecast()
    #app.run(debug=True)