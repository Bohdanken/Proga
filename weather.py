import requests
import datetime
import pytz
import pandas

API_KEY = ""
with open("weather_api_key.txt", "r") as f:
    API_KEY = f.read()
    f.close()




URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

"""
Input: 
timestamp: "2022-06-20T17:03:23.379152+02:00"  (ISO format, string)
Output: {"2022-06-20": ["18:00:00", "19:00:00", "20:00:00", "21:00:00", "22:00:00", "23:00:00"], 
         "2022-06-21": ["00:00:00", "01:00:00", "02:00:00", "03:00:00", "04:00:00", "05:00:00"]}
"""
def next_24_timestamps(timestamp):
    dic = dict()
    minute0 = timestamp[14:16]
    if int(minute0) < 30:
        timestamp = datetime.datetime.fromisoformat(timestamp)
        timestamp += datetime.timedelta(hours=1)
    ts = pandas.Timestamp(timestamp)
    ts = ts.round(freq = "H")
    stamp1 = ts.to_pydatetime()
    for i in range(24):
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
Output: list of dictionaries, where keys are the headers of the csv file "all_weather_by_hour.csv"
"""
def get_weather_forecast(timestamp, town):
    output = []
    timepoints = next_24_timestamps(timestamp)
    columns = ["city_latitude","city_longitude","city_resolvedAddress","city_address","city_timezone","city_tzoffset","day_datetime",\
               "day_datetimeEpoch","day_tempmax","day_tempmin","day_temp","day_feelslikemax","day_feelslikemin","day_feelslike","day_dew",\
                "day_humidity","day_precip","day_precipprob","day_precipcover","day_snow","day_snowdepth","day_windgust","day_windspeed",\
                "day_winddir","day_pressure","day_cloudcover","day_visibility","day_solarradiation","day_solarenergy","day_uvindex"\
               ,"day_severerisk","day_sunrise","day_sunriseEpoch","day_sunset","day_sunsetEpoch","day_moonphase","day_conditions",\
                "day_description","day_icon","day_source","day_preciptype","day_stations","hour_datetime","hour_datetimeEpoch",\
                "hour_temp","hour_feelslike","hour_humidity","hour_dew","hour_precip","hour_precipprob","hour_snow","hour_snowdepth"\
                ,"hour_preciptype","hour_windgust","hour_windspeed","hour_winddir","hour_pressure","hour_visibility","hour_cloudcover",\
                "hour_solarradiation","hour_solarenergy","hour_uvindex","hour_severerisk","hour_conditions","hour_icon","hour_source","hour_stations"]
    
    for day in timepoints.keys():
        responce = requests.get(URL+town+"/"+day+"T"+timepoints[day][0]+"?key="+API_KEY+"&unitGroup=metric")
        #print(URL+town+"/"+day+"T"+timepoints[day][0]+"?key="+API_KEY+"&unitGroup=metric")
        responce = responce.json()
    
        for hour in responce["days"][0]["hours"]:
            if hour["datetime"] in timepoints[day]:
                newdict = dict()
                for key in columns:
                    if key[:4] == "city":
                        newdict[key] = responce[key[5:]]
                    elif key[:3] == "day":
                        if key[4:] == "severerisk":
                            try:
                                newdict[key] = responce["days"][0][key[4:]]
                            except:
                                newdict[key] = 10.782214632864836 # average from the .csv
                        else:
                            newdict[key] = responce["days"][0][key[4:]]
                    else:
                        if key[5:] == "severerisk":
                            try:
                                newdict[key] = hour[key[5:]]
                            except:
                                newdict[key] = 10.782214632864836 # average from the .csv
                        else:
                            newdict[key] = hour[key[5:]]
                output.append(newdict)
                


    return output
"""
print("The names and the order of keys in the returned list's dictionaries")
print()
output = get_weather_forecast(datetime.datetime.now(pytz.timezone("Europe/Kyiv")).isoformat(), "Poltava,UA")
for i in output[0].keys():
    print(i)
print(len(output[0].keys()))
"""
"""
Input: list of dictionaries, all with the same keys
Output: list of Pandas's DataFrames
"""
def vectorize(data):
    df = pandas.DataFrame.from_dict(data)
    df["city_address"] = df["city_address"].apply(lambda x: str(x))
    df["city_timezone"] = df["city_timezone"].apply(lambda x: str(x))
    return df