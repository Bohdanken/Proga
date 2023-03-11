import requests
import datetime
import pytz
import pandas

API_KEY = "DNEUKTGQWMW3SDPAH7Z75Y9F8"
URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

"""
Input: 
timestamp: "2022-06-20T17:03:23.379152+02:00"  (ISO format, string)

Output: {"2022-06-20": ["18:00:00", "19:00:00", "20:00:00", "21:00:00", "22:00:00", "23:00:00"], 
         "2022-06-21": ["00:00:00", "01:00:00", "02:00:00", "03:00:00", "04:00:00", "05:00:00"]}
"""
def next_12_timestamps(timestamp):
    dic = dict()
    minute0 = timestamp[14:16]
    if int(minute0) < 30:
        timestamp = datetime.datetime.fromisoformat(timestamp)
        timestamp += datetime.timedelta(hours=1)
    ts = pandas.Timestamp(timestamp)
    ts = ts.round(freq = "H")
    stamp1 = ts.to_pydatetime()
    for i in range(12):
        if stamp1.isoformat()[:10] not in dic.keys():
            dic[stamp1.isoformat()[:10]] = [stamp1.isoformat()[11:19]]
        else:
            dic[stamp1.isoformat()[:10]].append(stamp1.isoformat()[11:19])
        stamp1 += datetime.timedelta(hours=1)
     
    return dic
    

"""
Input: 
timestamp: "2022-06-20T17:03:23.379152+02:00"  (ISO format, string). USE GMT+2 (Europe/Kyiv)!
town: "Poltava,UA"

Output: List of 12 JSONs for each hour after "timestamp"
"""
def get_weather_forecast(timestamp, town):
    output = dict()
    timepoints = next_12_timestamps(timestamp)
    
    for day in timepoints.keys():
        responce = requests.get(URL+town+"/"+day+"T"+timepoints[day][0]+"?key="+API_KEY)
        
        responce = responce.json()
        output[day] = []
        for hour in responce["days"][0]["hours"]:
            if hour["datetime"] in timepoints[day]:
                output[day].append(hour)

    return output
