import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
import constants
import time


API_KEY="08e8255b8852585a71b45fea6b96a1dc-d138dc156c5f4821f1477211546b32ce"
ACCOUNT_ID="101-011-20107572-001"
OANDA_URL = "https://api-fxpractice.oanda.com/v3"
SECURE_HEADER = {'Authorization': f'Bearer {API_KEY}'}
url = f"{OANDA_URL}/instruments/EUR_USD/candles"
 

def datetime_to_string(value, format='%Y-%m-%d %H:%M:%S'):
    return value.strftime(format)


def fetch_data():

    session = requests.Session()

    count = constants.COUNT

    list_of_data = []

    granularity_dict = {"M1":1,
    "M5":5,
    "M15":15,
    "H1":60,
    "H4":240,
    "D": 1440
    }

    data_time = []
    data_date = []
    data_high=[]
    data_low = []
    data_open = []
    data_close = []
    data_volume=[]
    flag = None
    date_time = []
    granularity_list = []
    is_complete = []



    for j in range(0,len(constants.GRANULARITY)):
        
        params = dict(
                count = count,
                granularity = constants.GRANULARITY[j],
                price = "MBA"
        )

        response = session.get(url,params=params, headers=SECURE_HEADER)
        data = response.json()['candles']

        for i in range(count):

            #get the date and time
            date = data[i]["time"]
            data_volume.append(int(data[i]["volume"]))
            is_complete.append(int(data[i]["complete"]))
            
            dt = parse(data[i]["time"])
            dt = datetime_to_string(dt)
            dt = dt.split()
            data_time.append(dt[1])
            timestamp = time.mktime(time.strptime(f'{dt[0]} {dt[1]}', '%Y-%m-%d %H:%M:%S'))
            date_time.append(float(timestamp))

            #get the mid high low close open
            mid = data[i]['mid']

            data_high.append(float(mid['h']))
            data_low.append(float(mid['l']))
            data_open.append(float(mid['o']))
            data_close.append(float(mid['c']))
            granularity_list.append(int(granularity_dict[constants.GRANULARITY[j]]))
            

        # data = pd.DataFrame(data)
        # print(data)

    preprocess_data = {
    'timestamp':date_time,
    'High':data_high,
    'Low':data_low,
    'Open':data_open,                  
    'Close':data_close,
    'Volume':data_volume,
    'Granularity':granularity_list,
    "Complete": is_complete
    }

    df = pd.DataFrame(preprocess_data)
    df.to_csv(f"gen_data.csv",index=False)
    return df
# a = np.array(preprocess_data)
# np.savetxt("foo.csv", a, delimiter=",")