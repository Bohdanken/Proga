
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import sep, remove
import pickle
import pandas as pd
from datetime import date, datetime, timedelta
import utils
import requests
from bs4 import BeautifulSoup
import re
import nltk


from num2words import num2words

import pickle
import weather
from sklearn import preprocessing
import scipy.sparse
import csv


def get_date_from_user():
    #return datetime.combine(date.today() - timedelta(days=2),datetime.min.time())
    today = datetime.combine(date.today(),datetime.min.time())
    
    query = ""
    while query == "":
        try:
            query = datetime.strptime(input("Enter a date (YYYY-MM-DD): "), "%Y-%m-%d")
        except:
            print("Wrong format! Try again")
   
    while query < datetime(2023, 1, 20) or query > today:
        print("Failure!")
        try:
            query = datetime.strptime(input("Enter a date (YYYY-MM-DD): "), "%Y-%m-%d")
        except:
            print("Wrong format! Try again")
    time_query = ""
    while time_query == "":
        try:
            time_query = datetime.strptime(input("Enter time (HH:MM): "), "%H:%M")
        except:
            print("Wrong format! Try again")
    
    return datetime.combine(query.date(),time_query.time())
    


    
  

def remove_names_and_date(page_html_text):
    parsed_html = BeautifulSoup(page_html_text, features="html.parser")
    p_lines = parsed_html.findAll('p')
    
    min_sentense_word_count = 13
    p_index = 0
    
    #find first long sentense
    for p_line in p_lines:
        
        strong_lines = p_line.findAll('strong')
        if not strong_lines:
            continue

        for s in strong_lines:
            if len(s.text.split(" ")) >= min_sentense_word_count:
                break
        else:
            p_index += 1
            continue
        break
    for i in range(0, p_index):
        page_html_text = page_html_text.replace(str(p_lines[i]), "")
        
    return page_html_text 

      



def get_region_from_user():
    city = input("Enter the name of the region's capital (for example: Poltava or Полтава): ")
    while ((city not in df_regions["center_city_ua"].values) and (city not in df_regions["center_city_en"].values)):
        print("This is not a region's capital in Ukraine. Try again!")
        city = input("Enter the name of the region's capital (for example: Poltava or Полтава): ")
    index = df_regions.index[df_regions["center_city_en"] == city].tolist()[0] if city in df_regions["center_city_en"].values else df_regions.index[df_regions["center_city_ua"] == city].tolist()[0]
    return df_regions.iloc[index] #.at["center_city_en"]




def merge_weather_isw_region(df_weather, df_isw, df_region):
    #print(df_isw.shape)
    fields_to_exlude = [
    "city_resolvedAddress", 
    "day_datetime",
    "day_datetimeEpoch",
    "hour_datetime",
    "hour_datetimeEpoch",
    "city",
    "region",
    "center_city_ua",
    "center_city_en",
    "isw_report_date",
    "isw_date_tomorrow_datetime",
    "isw_text_main",
    "isw_keywords",
    "isw_data_lemmatized"
    ]
    df_isw["report_date"] = df_isw["date"].apply(lambda x: datetime.combine(x, datetime.min.time()))
    df_isw["date_tomorrow_datetime"] = df_isw["report_date"].apply(lambda x: (x + timedelta(days=1)))
    
    df_isw_short = df_isw[["report_date", "date_tomorrow_datetime", "keywords", "text_main", "data_lemmatized",]]
    weather_region_df = pd.merge(left=df_weather,right=df_region, left_on="city", right_on="center_city_ua")
    df_isw_short = df_isw_short.copy().add_prefix('isw_')
    df = weather_region_df.merge(df_isw_short,
                            how = "left",
                            left_on = "day_datetime", 
                            right_on = "isw_date_tomorrow_datetime")
    df['day_datetime'] = pd.to_datetime(df['day_datetime'])
    df_v2 = df.drop(fields_to_exlude, axis=1)
    short_df_region = df_region[["region_alt", "region_id"]]
    df_v2 = df_v2.merge(short_df_region, 
                            how = "left",   
                            left_on = "region_alt", 
                            right_on = "region_alt")
    df_v2["hour_conditions"] = df_v2["hour_conditions"].apply(lambda x: x.split(",")[0])
    label_encoder = preprocessing.LabelEncoder()
    df_v2["hour_conditions_id"] = label_encoder.fit_transform(df_v2["hour_conditions"])
    tmp_fields_to_exlude = [
    "city_latitude",          
    "city_longitude",         
    "city_address",            
    "city_timezone",         
    "city_tzoffset",
    "day_feelslike",
    "day_feelslikemax",
    "day_feelslikemin",
    "day_sunrise", 
    "day_sunset", 
    "day_precipprob",
    "day_snow",
    "day_snowdepth",
    "day_windgust",
    "day_windspeed",
    "day_winddir",
    "day_pressure",
    "day_cloudcover",
    "day_visibility",
    "day_severerisk",
    "day_sunriseEpoch",
    "day_sunsetEpoch",
    "day_conditions",
    "day_description",
    "day_icon",
    "day_source",
    "day_preciptype",
    "day_stations",
    "hour_feelslike",
    "hour_preciptype",
    "hour_conditions",
    "region_alt",
    "hour_solarenergy",
    "hour_icon",
    "hour_source",
    "hour_stations"
    ]
    
    df_work_v3 = df_v2.drop(tmp_fields_to_exlude, axis=1).fillna(method="ffill")
    
    df_work_v3["region_id_x"] = df_work_v3["region_id_x"].apply(lambda x: int(x))
    df_work_v3["region_id_y"] = df_work_v3["region_id_y"].apply(lambda x: int(x))
    default_values = csv.DictReader(open("data"+sep+"weather_alarms_regions"+sep+"all_weather_by_hour_v2_mean_values.csv"))
    

    
    
    for i in default_values:
        df_work_v3.fillna(value=i, inplace=True)
        
    df_work_v3['day_solarradiation'] = df_work_v3['day_solarradiation'].astype('float64')
    df_work_v3['day_solarenergy'] = df_work_v3['day_solarenergy'].astype('float64')
    df_work_v3['day_uvindex'] = df_work_v3['day_uvindex'].astype('float64')
    df_work_v3['hour_visibility'] = df_work_v3['hour_visibility'].astype('float64')
    df_work_v3['hour_solarradiation'] = df_work_v3['hour_solarradiation'].astype('float64')
    df_work_v3['hour_uvindex'] = df_work_v3['hour_uvindex'].astype('float64')
    
    df_work_v3_csr = scipy.sparse.csr_matrix(df_work_v3.values)  
    
    
    word_count_vector = utils.cv.transform(df['isw_data_lemmatized'].values.astype('U'))
    
    with open(f"model"+sep+"word_count_vector_calculated.pkl", 'wb') as handle:
        pickle.dump(word_count_vector, handle)
    tfidf_vector = utils.tfidf.transform(word_count_vector)
    
    df_all_features = scipy.sparse.hstack((df_work_v3_csr, tfidf_vector), format='csr')
    return df_all_features

    

    




if __name__ == "__main__":
    label_encoder = pickle.load(open("model"+sep+"weather_conditions_label_encoder.pkl", "rb"))
    df_regions = pd.read_csv("data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")
    #print(357)
    chosen_date = get_date_from_user()
    #print(359)
    text_df = utils.get_text_df(chosen_date - timedelta(days=1))
    #print(361)
    text_df["date"] = chosen_date.date() - timedelta(days=1)
    region_df = get_region_from_user()
    #print(364)
    weather_forecast_df = weather.vectorize(weather.get_weather_forecast(chosen_date.isoformat(), region_df.at["center_city_en"]+",UA"))
    #print(366)
    weather_forecast_df["day_datetime"] = pd.to_datetime(weather_forecast_df["day_datetime"])
    weather_forecast_df["city"] = weather_forecast_df["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
    weather_forecast_df["city"] = weather_forecast_df["city"].replace("Хмельницька область", "Хмельницький")
    input_df = merge_weather_isw_region(weather_forecast_df, text_df, pd.DataFrame(region_df).transpose())
    #print(371)

    with open("model"+sep+"8_random_forest_v2.pkl", "rb") as modelfile:
        clf = pickle.load(modelfile, encoding="utf-8")
        schedule = clf.predict(input_df)


    print("Prediction for the next 24 hours. \"0\" means there will be no alarm, \"1\" - there will be:")
    print(schedule)

    

    
    
    



    




