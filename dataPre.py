# Data Collection, Preprocessing, and Merging
# Using multiple datasets, creating a merged dataset of university data across PA
# Goal of project: classify best university for a user based on inputs

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Data:
    def collect(self):
        data1 = pd.read_csv('data/Supply_Demand_Gap_Analysis_2014-2024_State_System_of_Higher_Education_20260403.csv') # https://data.pa.gov/Post-Secondary-Education/Supply-Demand-Gap-Analysis-2014-2024-State-System-/n3sx-ckrr/about_data
        # data2 may not work super well with the other data, but we can get some info from it
        data2 = pd.read_csv('data/PA_Higher_Education_Institutions_by_Sector_Current_Statewide_Education_20260404.csv') # https://data.pa.gov/Post-Secondary-Education/PA-Higher-Education-Institutions-by-Sector-Current/8u5h-xvus/about_data
        # Have to still research -> data3 = https://data.pa.gov/Education/Education-Names-and-Addresses-EdNA-Post-Secondary-/dtvt-jb9p/about_data
        print(data1.head())
        print(data2.head())

        return data1, data2

    def merge(self):
        data1, data2 = self.collect()
        dataset = pd.concat([data1, data2], axis=1)
        # Need to add how they are concatenated, filtering, etc.
        return dataset

    def split(self):
        dataset = self.merge()
        x = dataset.drop(columns=['name of insitution col'])
        y = dataset['name of insitution col']
        x = LabelEncoder().fit_transform("add categorical cols here")
        x = StandardScaler().fit_transform(x)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

        return train_x, test_x, train_y, test_y


data = Data()
data.collect()