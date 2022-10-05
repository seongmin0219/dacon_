import pandas as pd


class DataLoader():

    def __init__(self):
        pass
    def load_data(self,DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df


