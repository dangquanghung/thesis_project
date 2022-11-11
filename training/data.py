import requests
import pandas as pd


class TimeSeriesData:
    def __init__(self) -> None:
        self.url = "https://api.thingspeak.com/channels/1576707/feeds.json?results=50000"
        
    def clean_data(self, df):
        # may need to implement when extended
        return df
    
    def get_data(self):
        r = requests.get(self.url)
        data = pd.DataFrame(r.json()['feeds'])
        data.columns = ["created_at", "entry_id", "LPG", "CO"]  # type: ignore
        data = self.clean_data(data)
        return data
    