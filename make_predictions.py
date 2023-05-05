import numpy as np
from flask import Flask, request

from os import sep
import pickle
import pandas as pd
import utils

import json
from flask import Response
import os
import re
from datetime import datetime, timedelta
from itertools import pairwise

import weather

from run import merge_weather_isw_region

app = Flask(__name__)
model_path = "model/tuning/logistic_regression_v11/"
model = "logistic_regression_v11_lbfgs_split3.pkl"
path_to_forecasts = "data/forecasts/"
time_as_path = 'data/forecasts/2023-04-24 2.json'

@app.route('/send_prediction', methods=['POST'])
def send_forecast():
    last_model_train_time = datetime.fromtimestamp(os.path.getmtime(f"{model_path}{model}")).strftime("%Y-%m-%d %H:%M")
    meta_information = "Slim Shady"
    data = request.json
    region = data['region']
    date_requested = data["date"]
    time_requested = data['time']
    date_time_requested = datetime.strptime(f"{date_requested} {time_requested}", "%Y-%m-%d %H")
    predictions=check_for_existing_alarm_prediction_and_calculate(date_time_requested)
    last_alarm_forecasts_time = datetime.fromtimestamp(os.path.getmtime(f"data/predicted_alarms/{date_requested} {time_requested}.json")).strftime("%Y-%m-%d %H:%M")
    return sendPredict(predictions, region, last_model_train_time, last_alarm_forecasts_time, meta_information)


def sendPredict(predictions,region,last_model_train_time,last_alarm_forecasts,meta_information):
    time_array = predictions['hour_datetime'].tolist()
    predictions=predictions.drop('hour_datetime',axis=1)
    with open("data/predictions/regions.pkl", "rb") as f:
        regions = pickle.load(f)
    predictions = predictions.T.values
    region_forecasts = {}
    for i in range(predictions.shape[0]):
        # Get the region name
        region_name = regions[i]
        # Get the row of predictions for this region
        region_predictions = predictions[i]
        # Convert the row of predictions to a list of boolean values
        region_forecast = region_predictions.astype(bool).tolist()
        # Build a dictionary for this region with the time strings as keys and the forecast values as values
        region_dict = dict(zip(time_array, region_forecast))
        # Add the region dictionary to the region forecasts dictionary
        region_forecasts[region_name] = region_dict
    if region not in regions:
        response_dict = {
            'last_model_train_time': last_model_train_time,
            'last_prediction_time': last_alarm_forecasts,
            'meta_information': meta_information,
            'regions_forecast': region_forecasts
        }
    else:
        forecast = region_forecasts[region]
        response_dict = {
            'last_model_train_time': last_model_train_time,
            'last_prediction_time': last_alarm_forecasts,
            'meta_information': meta_information,
            'regions_forecast': forecast
        }
    json_response = json.dumps(response_dict)
    return Response(json_response, status=200, mimetype='application/json')



def check_for_existing_alarm_prediction_and_calculate(date_time_requested):
    date_requested=date_time_requested.date()
    time_requested=date_time_requested.hour
    predictions=None
    missing=check_prediction_exist("data/predicted_alarms/",date_requested,time_requested)
    if missing is None:
        predictions=read_files_to_df("data/predicted_alarms/",date_requested,time_requested)
    else:
        last_forecasts=last_forecast1("data/forecasts/",date_requested,time_requested)
        forecast=calculate_forecast(last_forecasts,date_time_requested)
        text_df = utils.get_text_df(date_time_requested - timedelta(days=1))
        text_df["date"] = date_time_requested.date() - timedelta(days=1)
        for miss in missing:
            calculate(forecast,miss,text_df)
        predictions = read_files_to_df("data/predicted_alarms/", date_requested, time_requested)
    return predictions

def read_files_to_df(folder_path, date, starting_hour):
    """
    Reads all 12 JSON files for the given date and starting hour into a single dataframe.
    """
    # Initialize an empty list to hold the dataframes for each hour
    dfs = []
    current_time = datetime.strptime(f"{date} {starting_hour}", "%Y-%m-%d %H")
    end_time = current_time + timedelta(hours=12)
    # Loop through the next 12 hours starting from the given starting hour
    while current_time < end_time:
        # Construct the file name for the current hour
        filename = os.path.join(folder_path, f"{current_time.date() } {current_time.hour}.json")
        # Read the JSON file into a dataframe
        df = pd.read_json(filename)
        # Append the dataframe to the list
        dfs.append(df)
        current_time += timedelta(hours=1)
    # Concatenate the dataframes into a single dataframe
    result = pd.concat(dfs)
    return result


def calculate_forecast(last_weather_forecasts,date_time_requested):
    date_time_requested_str=str(date_time_requested.date()) + " " + str(date_time_requested.hour)
    last_weather_forecast=None
    if (len(last_weather_forecasts) == 0):
        get_forecast_for_all(date_time_requested)
        last_weather_forecast = pd.read_json(f"data/forecasts/{date_time_requested_str}.json")
    elif (len(last_weather_forecasts) == 1):
        last_weather_forecast = pd.read_json(f"data/forecasts/{last_weather_forecasts[0]}.json")
    elif (len(last_weather_forecasts) == 2):
        last_weather_forecast1 = pd.read_json(f"data/forecasts/{last_weather_forecasts[0]}.json")
        last_weather_forecast2 = pd.read_json(f"data/forecasts/{last_weather_forecasts[1]}.json")
        last_weather_forecast = combine_2_dates(last_weather_forecast1, last_weather_forecast2, date_time_requested,
                                                on="day_time_compare")
    return last_weather_forecast

@app.route('/calculate', methods=['POST'])
def calculate(weather_forecast_df_all_regions, time_used,text_df):
    hours=1
    cutoff_time = time_used + timedelta(hours=hours)
    schedules = []
    df_regions = pd.read_csv("data" + sep + "weather_alarms_regions" + sep + "regions.csv", sep=",")
    groups = weather_forecast_df_all_regions.groupby(weather_forecast_df_all_regions.index // 24)
    regions = df_regions['center_city_en'].tolist()
    for index, forecast in groups:
        forecast["day_time_compare"] = pd.to_datetime(
            forecast['day_datetime'].str.split().str[0] + ' ' + forecast['hour_datetime'])
        forecast["day_datetime"] = pd.to_datetime(forecast["day_datetime"])
        filter1 = forecast["day_time_compare"] < cutoff_time
        filter2 = forecast['day_time_compare'] >= time_used
        filter = filter1 & filter2
        forecast = forecast.drop("day_time_compare", axis=1)
        filtered_forecast = forecast[filter]
        filtered_forecast.reset_index(drop=True)
        filtered_forecast["city"] = filtered_forecast["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
        filtered_forecast["city"] = filtered_forecast["city"].replace("Хмельницька область", "Хмельницький")
        input_df = merge_weather_isw_region(filtered_forecast, text_df, df_regions)

        with open(f"{model_path}{model}", "rb") as modelfile:
            clf = pickle.load(modelfile, encoding="utf-8")
            schedule = clf.predict(input_df)
            schedules.extend(schedule.tolist())
    predictions_np = np.reshape(schedules, (25, hours))
    predictions_df = pd.DataFrame(predictions_np).T
    hours_df = pd.date_range(start=time_used, freq='H', periods=len(predictions_df))
    # convert the range of hours to a Pandas Series
    hours_series = pd.Series(hours_df.strftime('%Y-%m-%d %H'))
    # add the new column to the DataFrame
    predictions_df['hour_datetime'] = hours_series
    time_str = str(time_used.date()) + " " + str(time_used.hour)
    # save predictions to file
    predictions_df.to_json(f'data/predicted_alarms/{time_str}.json')
    with open("data/predictions/regions.pkl", "wb") as f:
        pickle.dump(regions, f)


def check_prediction_exist(folder_path, date, starting_hour):
    # Initialize a list to hold the datetimes of the missing files
    missing_files = []
    current_time = datetime.strptime(f"{date} {starting_hour}","%Y-%m-%d %H")
    end_time=current_time+timedelta(hours=12)
    # Loop through the next 12 hours starting from the given starting hour
    while current_time < end_time:
        # Construct the file name for the current hour
        filename = os.path.join(folder_path, f"{current_time.date()} {current_time.hour}.json")
        # Check if the file exists
        if not os.path.isfile(filename):
            # If the file is missing, append the datetime to the missing_files list
            missing_files.append(datetime.strptime(f"{current_time.date()} {current_time.hour}", "%Y-%m-%d %H"))
        current_time+=timedelta(hours=1)
    # If any files are missing, return the list of missing datetimes
    if missing_files:
        return missing_files
    # If we get here, all files exist
    return None

def last_forecast1(folder_path, date_requested, time_requested):
    file_names = os.listdir(folder_path)
    datetime_requested = datetime.strptime(f"{date_requested} {time_requested}", "%Y-%m-%d %H")
    # Define a regular expression pattern to match the prediction time in the file names
    pattern = r'^\d{4}-\d{2}-\d{2} \d{1,2}\..*$'

    # Initialize a variable to hold the closest prediction times
    closest_prediction_times = []
    # Loop through the file names and find the closest prediction times
    for file_name in file_names:
        # Check if the file name matches the pattern
        match = re.match(pattern, file_name)
        if match:
            # Extract the prediction time from the matched pattern
            prediction_time_str = re.sub("\\..*", "", file_name)
            # Convert the prediction time string to a datetime object
            prediction_time = datetime.strptime(prediction_time_str, "%Y-%m-%d %H")
            # Check if this prediction time is within 24 hours of the requested time
            if abs(prediction_time - datetime_requested) <= timedelta(hours=23):
                closest_prediction_times.append(prediction_time)
    # Sort the closest prediction times by their distance to the requested time
    closest_prediction_times.sort(key=lambda x: datetime_requested - x, reverse=True)
    last_predictions = []
    last_prediction=datetime.fromtimestamp(0)
    if len(closest_prediction_times) == 0:
        return []
    for i in range(0, len(closest_prediction_times)):
        if  datetime_requested-timedelta(hours=12)<=closest_prediction_times[i] <= datetime_requested:
            answer = str(closest_prediction_times[i].date()) + " " + str(closest_prediction_times[i].hour)
            return [answer]
        elif datetime_requested > closest_prediction_times[i]:
            current_pair = find_timespan(closest_prediction_times[i], closest_prediction_times[i + 1:],
                                         datetime_requested)
            if current_pair:
                last_predictions = current_pair
            elif closest_prediction_times[i]>last_prediction:
                last_prediction=closest_prediction_times[i]
    if not last_predictions:
        last_predictions=[last_prediction,None,None]
    # Return the list of filenames for the closest prediction times (up to 2)
    return last_predictions


def find_timespan(current_date_time, date_times, chosen_time):
    end_time = current_date_time + timedelta(hours=24)
    current_date_time_str = str(current_date_time.date()) + " " + str(current_date_time.hour)
    pair = []
    for dt in date_times:
        if dt <= end_time and dt > chosen_time:
            answer = str(dt.date()) + " " + str(dt.hour)
            pair = [current_date_time_str, answer]
    return pair


def combine_2_dates(df1, df2, requested_time, on):
    # Convert hour_datetime to datetime format
    df1["day_time_compare"] = pd.to_datetime(
        df1['day_datetime'].str.split().str[0] + ' ' + df1['hour_datetime'])
    df2["day_time_compare"] = pd.to_datetime(
        df2['day_datetime'].str.split().str[0] + ' ' + df2['hour_datetime'])
    # Merge the two dataframes
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df["day_time_compare_str"]=merged_df["day_time_compare"].dt.strftime('%Y-%m-%d %H:%M:%S')
    merged_df = merged_df.groupby('city_address',sort=False).apply(
        lambda x: x.drop_duplicates(subset='day_time_compare_str', keep='first')).reset_index(drop=True)
    # Filter the merged dataframe to get the desired time span
    filtered_df = merged_df[(merged_df['day_time_compare'] >= requested_time) & (
            merged_df['day_time_compare'] < requested_time + timedelta(hours=24))]
    filtered_df = filtered_df.drop(["day_time_compare_str", "day_time_compare"], axis=1)
    filtered_df=filtered_df.reset_index(drop=True)
    return filtered_df


@app.route('/update', methods=['POST'])
def get_forecast_for_all(chosen_date):
    df_regions = pd.read_csv("data" + sep + "weather_alarms_regions" + sep + "regions.csv", sep=",")
    df_forecast = pd.DataFrame()
    for i, region in enumerate(df_regions["center_city_en"]):
        weather_forecast_df = weather.vectorize(weather.get_weather_forecast((chosen_date-timedelta(hours=1)).isoformat(), region + ",UA"))
        df_forecast = pd.concat([df_forecast, weather_forecast_df], axis=0)
    df_forecast = df_forecast.reset_index(drop=True)
    print(len(df_forecast["city_resolvedAddress"].unique()))
    global time
    time = chosen_date
    global time_as_path
    time_as_path = path_to_forecasts + str(str(chosen_date.date()) + " " + str(chosen_date.hour) + ".json", )
    df_forecast.to_json(path_or_buf=time_as_path)

# print(last_forecast("data/forecasts", "2023-04-24", "15"))
# get_forecast_for_all(datetime.strptime("2023-04-24 15", "%Y-%m-%d %H" ))

time=datetime.now()
predictions=check_for_existing_alarm_prediction_and_calculate(time)

if __name__ == '__main__':
    app.run(debug=True)
