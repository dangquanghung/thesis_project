import requests
import pandas as pd


class TimeSeriesData:
    def __init__(self) -> None:
        self.url = "https://api.thingspeak.com/channels/1576707/feeds.json?results=50000"
        self.file = "data_sync.csv"
        
    def clean_data(self, df):
        # may need to implement when extended
        return df
    
    def get_data(self):
        r = requests.get(self.url)
        data = pd.DataFrame(r.json()['feeds'])
        data.columns = ["created_at", "entry_id", "LPG", "CO"]  # type: ignore
        data = self.clean_data(data)
        return data
    
    def get_data2(self):
        r = requests.get(self.file)
        data = pd.read_csv('data_sync.csv', header=None, names=['time', 'LPG', 'CO'])
        data['created_at'] = pd.to_datetime( data['time'], unit='ms')
        data.columns = ["created_at", "entry_id", "LPG", "CO"]
        data = self.clean_data(data)
        return data
    