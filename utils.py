import pickle
from os import sep, remove
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import re
import nltk
import unicodedata
import string
from num2words import num2words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import json


# tfidf = pickle.load(open("C:/Users/fadim/python_course/Proga/Proga/model/tfidf_transformer_v1.pkl", "rb"))
cv = pickle.load(open("model" + sep + "count_vectorizer_v1.pkl", "rb"))
tfidf = pickle.load(open("model" + sep + "tfidf_transformer_v1.pkl", "rb"))


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def save_page(url, file_name):
    page = requests.get(url)
    # with open('D:\Python project\\'+file_name +".html", "w") as f:
    # f.write(url+'\n')
    with open(file_name + ".html", "wb+") as f:
        f.write(page.content)
        f.close()
        return file_name + ".html"


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}

    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def text_processing(file_name):
    d = {}
    with open(file_name, "r", encoding='utf-8') as cfile:
        # url=cfile.readline()
        parsed_html = BeautifulSoup(cfile.read(), features="html.parser")
        title = parsed_html.head. find('title').text
        textS_title = parsed_html.body.find(
            'h1', attrs={"id": 'page-title'}).text
        text_main = parsed_html.body.find('div', attrs={
                                          'class': 'field field-name-body field-type-text-with-summary field-label-hidden'})
        d = {"date": date,
             # "url":url,
             "title": title,
             "text_title": textS_title,
             "text_main": text_main}
    cfile.close()
    pd.DataFrame(d, index=[0]).to_csv("temp.csv")
    return "temp.csv"


def remove_names_and_date(page_html_text):
    parsed_html = BeautifulSoup(page_html_text, features="html.parser")
    p_lines = parsed_html.findAll('p')

    min_sentense_word_count = 13
    p_index = 0

    # find first long sentense
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
    # CHARS_TO_REMOVE=['\r','\n']
    return result.replace('\r', ' ').replace('\n', ' ')


def remove_any_punct(q):
    return q.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))


def remove_one_letter_word(data):
    word_tokens = nltk.word_tokenize(str(data))
    new_text = ""
    for w in word_tokens:
        if len(w) == 1:
            continue

        new_text = new_text + " " + w
    return new_text


def convert_numbers(data):
    tokens = nltk.word_tokenize(str(data))
    result = ""
    for w in tokens:
        if w.isdigit():
            if (int(w) > 10000000000):
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
        results[feature_vals[idx]] = score_vals[idx]
    return results


def conver_doc_to_vector(doc, cv, tfidf):
    feature_names = cv.get_feature_names_out()
    top_n = 100
    tf_idf_vector = tfidf.transform(cv.transform([doc]))

    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, top_n)

    return keywords


def get_text_df(chosen_date):
    html_file_name = save_page("https://understandingwar.org/backgrounder/russian-offensive-campaign-assessment-" +
                               str(chosen_date.strftime("%B"))+"-"+str(chosen_date.day)+"-2023", str(chosen_date.date()))

    divided_text_csv = text_processing(html_file_name)
    remove(html_file_name)
    divided_text = pd.read_csv(divided_text_csv)
    remove(divided_text_csv)
    divided_text['main_html_v2'] = divided_text['text_main'].apply(
        lambda x: remove_names_and_date(x))

    pattern = "\[(\d+)\]"
    divided_text['main_html_v3'] = divided_text['main_html_v2'].apply(
        lambda x: re.sub(pattern, "", x))
    divided_text['main_html_v4'] = divided_text['main_html_v3'].apply(
        lambda x: BeautifulSoup(x, features="html.parser").text)
    divided_text['main_html_v5'] = divided_text['main_html_v4'].apply(
        lambda x: re.sub(r'http(\S+.*\s)', "", x))
    divided_text['main_html_v6'] = divided_text['main_html_v5'].apply(
        lambda x: re.sub(r'(©2022|©2023|2022|2023)', "", x))
    divided_text['main_html_v7'] = divided_text['main_html_v6'].apply(
        lambda x: re.sub(r'\n.{5,15}\d:\d.{0,9}\n', "", x))
    divided_text['main_html_v8'] = divided_text['main_html_v7'].apply(lambda x: re.sub(
        'Appendix A – Satellite Imagery(.|\n)+\.', "", x)).apply(lambda x: re.sub('Click here to expand the map below.', "", x))
    divided_text = divided_text.drop(['Unnamed: 0', 'main_html_v2', 'main_html_v3',
                                     'main_html_v4', 'main_html_v5', 'main_html_v6', 'main_html_v7'], axis=1)
    nltk.download()
    words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

    divided_text['main_html'] = divided_text['main_html_v8'].apply(
        lambda x: x.lower())
    divided_text['main_html1'] = divided_text['main_html'].apply(
        lambda x: remove_special_characters(x))
    divided_text['main_html2'] = divided_text['main_html1'].apply(
        lambda x: remove_any_punct(x))
    divided_text['main_html3'] = divided_text['main_html2'].apply(
        lambda x: remove_one_letter_word(x))
    nltk.download('wordnet')
    divided_text['main_html4'] = divided_text['main_html3'].apply(
        lambda x: convert_numbers(x)).apply(lambda x: remove_stopwords(x))
    divided_text['data_stemmed'] = divided_text['main_html4'].apply(
        lambda x: stemming(x))
    divided_text['data_lemmatized'] = divided_text['main_html4'].apply(
        lambda x: lemmatizing(x))
    # divided_text['data_lemmatized'].to_csv("data_lemmatized.csv")
    docs = divided_text['data_lemmatized'].tolist()
    cv = CountVectorizer(min_df=0.98, max_df=2)
    word_count_vector = cv.fit_transform(docs)
    """
    with open("./model/count_vectorizer_v1.pkl", 'wb') as handle:
        pickle.dump(cv, handle)
    """
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True,)
    tfidf_transformer.fit(word_count_vector)
    """
    with open("model/tfidf_transformer_v1.pkl", 'wb') as handle:
        pickle.dump(tfidf_transformer, handle)
        """
    df_idf = pd.DataFrame(tfidf_transformer.idf_,
                          index=cv.get_feature_names_out(), columns=["idf_weights"])
    df_idf.sort_values(by=['idf_weights'])
    tf_idf_vector = tfidf_transformer.transform(word_count_vector)
    tfidf = pickle.load(open("model/tfidf_transformer_v1.pkl", "rb"))
    cv = pickle.load(open("model/count_vectorizer_v1.pkl", "rb"))
    feature_names = cv.get_feature_names_out()
    divided_text['keywords'] = divided_text['data_lemmatized'].apply(
        lambda x: conver_doc_to_vector(x))  # (x, cv, tfidf)
    return divided_text


def conver_doc_to_vector(doc):
    feature_names = cv.get_feature_names_out()
    top_n = 100
    tf_idf_vector = tfidf.transform(cv.transform([doc]))

    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, top_n)

    return keywords


def write_mean_values():
    list_of_columns_needed = ["day_tempmax",
                              "day_tempmin",
                              "day_temp",
                              "day_dew",
                              "day_humidity",
                              "day_precip",
                              "day_precipcover",
                              "day_solarradiation",
                              "day_solarenergy",
                              "day_uvindex",
                              "day_moonphase",
                              "hour_temp",
                              "hour_humidity",
                              "hour_dew",
                              "hour_precip",
                              "hour_precipprob",
                              "hour_snow",
                              "hour_snowdepth",
                              "hour_windgust",
                              "hour_windspeed",
                              "hour_winddir",
                              "hour_pressure",
                              "hour_visibility",
                              "hour_cloudcover",
                              "hour_solarradiation",
                              "hour_uvindex",
                              "hour_severerisk"]
    output = dict()
    df_all_weather = pd.read_csv(
        "data"+sep+"weather_alarms_regions"+sep+"all_weather_by_hour_v2.csv", sep=",")
    for i in df_all_weather.columns.values.tolist():
        if i in list_of_columns_needed:
            output[i] = df_all_weather[i].mean()
    pd.DataFrame.from_dict([output]).to_csv("data"+sep+"weather_alarms_regions" +
                                            sep+"all_weather_by_hour_v2_mean_values.csv", sep=',', encoding='utf-8')

# write_mean_values()
