
import pandas 
import datetime
from os import listdir
import requests

list_of_dates = pandas.date_range(datetime.date(2022, 2, 24), datetime.date(2023, 1, 20),freq='d')
list_of_dates = [str(i)[:10] for i in list_of_dates]
list_of_files = listdir("fff")
list_of_files = [i[:10] for i in list_of_files]
# print(set(list_of_dates)-set(list_of_files))
# output: {'2022-07-11', '2022-08-12', '2023-01-01', '2022-11-24', '2022-12-25', '2022-02-24', '2022-05-05'}
#               pdf            link         missing     missing         missing         link        link
#
# https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-august-12-0
# Note: ISW and CTP will not publish a campaign assessment (or maps) tomorrow, January 1, in observance of the New Year's Holiday
# After 280 consecutive days of reporting on the Russian invasion of Ukraine, ISW and CTP will not publish a campaign assessment (or maps) tomorrow, November 24. Coverage will resume Friday, November 25. 
# Note: ISW and CTP will not publish a campaign assessment (or maps) tomorrow, December 25, in observance of the Christmas holiday.
# https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-initial-russian-offensive-campaign-assessment
# https://www.understandingwar.org/backgrounder/russian-campaign-assessment-may-5
def save_page(url, file_name):
    page = requests.get(url)
    #with open('D:\Python project\\'+file_name +".html", "w") as f:
       # f.write(url+'\n')
    with open('fff\\'+file_name +".html", "wb+") as f:
        f.write(page.content)

"""
save_page("https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-august-12-0", "2022-08-12")
save_page("https://www.understandingwar.org/backgrounder/russia-ukraine-warning-update-initial-russian-offensive-campaign-assessment", "2022-02-24")
save_page("https://www.understandingwar.org/backgrounder/russian-campaign-assessment-may-5","2022-05-05")
"""

