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
import unicodedata
import string
from num2words import num2words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pickle
import weather
from sklearn import preprocessing
import scipy.sparse
MODEL_FOLDER="model"

def get_date_from_user():
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

def save_page(url, file_name):
    page = requests.get(url)
    #with open('D:\Python project\\'+file_name +".html", "w") as f:
       # f.write(url+'\n')
    with open(file_name +".html", "wb+") as f:
        f.write(page.content)
        f.close()
        return file_name +".html"
    
def text_processing(file_name):
    d={}
    with open(file_name, "r",encoding='utf-8') as cfile:
        #url=cfile.readline()
        parsed_html= BeautifulSoup(cfile.read(), features="html.parser")
        title = parsed_html.head. find('title').text
        textS_title = parsed_html.body.find('h1', attrs={"id":'page-title'}).text
        text_main = parsed_html.body.find('div', attrs={'class': 'field field-name-body field-type-text-with-summary field-label-hidden'})
        d={"date":date,
           #"url":url,
           "title": title,
           "text_title":textS_title,
           "text_main":text_main}
    cfile.close()
    pd.DataFrame(d,index=[0]).to_csv("temp.csv")
    return "temp.csv"   

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

def remove_special_characters(data):
    result = unicodedata.normalize("NFKD", data)
    #CHARS_TO_REMOVE=['\r','\n']
    return result.replace('\r', ' ').replace('\n', ' ')      

def remove_any_punct(q):
    return q.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))    

def remove_one_letter_word (data):
    word_tokens = nltk.word_tokenize(str(data))
    new_text = ""
    for w in word_tokens:
        if len(w) == 1 :
            continue

        new_text= new_text + " " + w
    return new_text

def convert_numbers(data):
    tokens = nltk.word_tokenize(str(data))
    result=""
    for w in tokens :
        if w.isdigit():
            if(int(w)>10000000000):
                continue
            w = remove_any_punct(num2words(w))
        result = result + ' ' + w
    return result

def remove_stopwords(data):
    stopwords = nltk.corpus.stopwords.words("english")
    word_tokens = nltk.word_tokenize(str(data))
    stop_stop_words = {"no", "not"}
    stop_words = set(stopwords) - stop_stop_words
    result = ""
    for w in word_tokens:
        if w not in stop_words:
            result = result + " " + w
    return result

def stemming(data):
    stemmer = nltk.PorterStemmer()
    tokens = nltk.word_tokenize(str(data))
    result = ""
    for w in tokens:
        result = result + " " + stemmer.stem(w)
    return result

def lemmatizing(data):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(str(data))
    result = ""
    for w in tokens:
        result = result + " " + lemmatizer.lemmatize(w)
    return result

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

def conver_doc_to_vector(doc, cv, tfidf):
    feature_names = cv.get_feature_names_out()
    top_n = 100
    tf_idf_vector = tfidf.transform(cv.transform([doc]))

    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, top_n)

    return keywords

def get_region_from_user():
    city = input("Enter the name of the region's capital (for example: Poltava or Полтава): ")
    while ((city not in df_regions["center_city_ua"].values) and (city not in df_regions["center_city_en"].values)):
        print("This is not a region's capital in Ukraine. Try again!")
        city = input("Enter the name of the region's capital (for example: Poltava or Полтава): ")
    index = df_regions.index[df_regions["center_city_en"] == city].tolist()[0] if city in df_regions["center_city_en"].values else df_regions.index[df_regions["center_city_ua"] == city].tolist()[0]
    return df_regions.iloc[index] #.at["center_city_en"]

def merge_weather_isw_region(df_weather, df_isw, df_region):
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
    print(df_work_v3["day_solarradiation"])
    df_work_v3["region_id_x"] = df_work_v3["region_id_x"].apply(lambda x: int(x))
    df_work_v3["region_id_y"] = df_work_v3["region_id_y"].apply(lambda x: int(x))

    

    df_work_v3_csr = scipy.sparse.csr_matrix(df_work_v3.values) # ERROR HERE 
    
    tfidf = pickle.load(open("model"+ sep+ "tfidf_transformer_v1.pkl", "rb"))
    cv = pickle.load(open("model"+ sep+ "count_vectorizer_v1.pkl", "rb"))
    word_count_vector = cv.transform(df['isw_data_lemmatized'].values.astype('U'))
    tfidf_vector = tfidf.transform(word_count_vector)
    df_all_features = scipy.sparse.hstack((df_work_v3_csr, tfidf_vector), format='csr')
    return df_all_features

    

    

def get_text_df(chosen_date):
    html_file_name = save_page("https://understandingwar.org/backgrounder/russian-offensive-campaign-assessment-"+str(chosen_date.strftime("%B"))+"-"+str(chosen_date.day)+"-2023", str(chosen_date.date()))
    
    divided_text_csv = text_processing(html_file_name)
    remove(html_file_name)
    divided_text = pd.read_csv(divided_text_csv)
    remove(divided_text_csv)
    divided_text['main_html_v2'] = divided_text['text_main'].apply(lambda x: remove_names_and_date(x))
    
    pattern = "\[(\d+)\]"
    divided_text['main_html_v3'] = divided_text['main_html_v2'].apply(lambda x: re.sub(pattern, "", x))
    divided_text['main_html_v4'] = divided_text['main_html_v3'].apply(lambda x: BeautifulSoup(x, features="html.parser").text)
    divided_text['main_html_v5'] = divided_text['main_html_v4'].apply(lambda x: re.sub(r'http(\S+.*\s)', "", x))
    divided_text['main_html_v6'] = divided_text['main_html_v5'].apply(lambda x: re.sub(r'(©2022|©2023|2022|2023)', "", x))
    divided_text['main_html_v7'] = divided_text['main_html_v6'].apply(lambda x: re.sub(r'\n.{5,15}\d:\d.{0,9}\n', "", x))
    divided_text['main_html_v8'] = divided_text['main_html_v7'].apply(lambda x: re.sub('Appendix A – Satellite Imagery(.|\n)+\.', "", x)).apply(lambda x: re.sub('Click here to expand the map below.', "", x))
    divided_text=divided_text.drop(['Unnamed: 0','main_html_v2','main_html_v3','main_html_v4','main_html_v5','main_html_v6','main_html_v7'],axis=1)
    nltk.download()
    words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
    
    divided_text['main_html'] = divided_text['main_html_v8'].apply(lambda x: x.lower())
    divided_text['main_html1'] = divided_text['main_html'].apply(lambda x: remove_special_characters(x))
    divided_text['main_html2'] = divided_text['main_html1'].apply(lambda x: remove_any_punct(x))
    divided_text['main_html3'] = divided_text['main_html2'].apply(lambda x: remove_one_letter_word(x))
    nltk.download('wordnet')
    divided_text['main_html4']=divided_text['main_html3'].apply(lambda x: convert_numbers(x)).apply(lambda x: remove_stopwords(x))
    divided_text['data_stemmed'] = divided_text['main_html4'].apply(lambda x: stemming(x))
    divided_text['data_lemmatized'] = divided_text['main_html4'].apply(lambda x: lemmatizing(x))
    docs = divided_text['data_lemmatized'].tolist()
    cv = CountVectorizer(min_df=0.98, max_df=2) #поміняв місцями мін та макс
    word_count_vector = cv.fit_transform(docs)
    with open("./model/count_vectorizer_v1.pkl", 'wb') as handle:
        pickle.dump(cv, handle)
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True,)
    tfidf_transformer.fit(word_count_vector)
    with open("model/tfidf_transformer_v1.pkl", 'wb') as handle:
        pickle.dump(tfidf_transformer, handle)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(), columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'])
    tf_idf_vector = tfidf_transformer.transform(word_count_vector)
    tfidf = pickle.load(open("model/tfidf_transformer_v1.pkl", "rb"))
    cv = pickle.load(open("model/count_vectorizer_v1.pkl", "rb"))
    feature_names = cv.get_feature_names_out()
    divided_text['keywords'] = divided_text['data_lemmatized'].apply(lambda x: conver_doc_to_vector(x, cv, tfidf))
    return divided_text

if __name__ == "__main__":
    label_encoder = pickle.load(open("model"+sep+"weather_conditions_label_encoder.pkl", "rb"))
    df_regions = pd.read_csv("data"+sep+"weather_alarms_regions"+sep+"regions.csv", sep=",")

    chosen_date = get_date_from_user()
    text_df = get_text_df(chosen_date - timedelta(days=1))
    text_df["date"] = chosen_date.date() - timedelta(days=1)
    region_df = get_region_from_user()
    weather_forecast_df = weather.vectorize(weather.get_weather_forecast(chosen_date.isoformat(), region_df.at["center_city_en"]+",UA"))
    weather_forecast_df["day_datetime"] = pd.to_datetime(weather_forecast_df["day_datetime"])
    weather_forecast_df["city"] = weather_forecast_df["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
    weather_forecast_df["city"] = weather_forecast_df["city"].replace("Хмельницька область", "Хмельницький")
    input_df = merge_weather_isw_region(weather_forecast_df, text_df, pd.DataFrame(region_df).transpose())
    print(input_df)
    file_path = "model" + sep + "8_logistic_regression_v3.pkl"
    clf=pickle.load(open(f"{MODEL_FOLDER}/8_logistic_regression_v3.pkl", "rb"))
    schedule = clf.predict(input_df)




    print(schedule)
    print("1")

    

    
    
    



    




