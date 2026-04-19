# Data Collection, Preprocessing, and Merging
# Using multiple datasets, creating a merged dataset of university data across PA
# Goal of project: classify best university for a user based on inputs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class Data:
    def collect(self):
        higherEdBySector = pd.read_csv('data/HigherEd_BySector.csv')
        higherEdBySector.name = "HigherEducationBySector"

        higherEdInstitutions = pd.read_csv('data/HigherEd_Insitutions.csv')
        higherEdInstitutions = higherEdInstitutions[higherEdInstitutions["County"] != "Out-of-State"]
        higherEdInstitutions.name = "HigherEducationInstitutions"
        #print(higherEdInstitutions['Category'].unique())

        cost = pd.read_csv('data/Data_4-19-2026---701.csv')
        cost.name = "Cost"

        """ PASSHE Data - unused
        totalHeadcount = pd.read_csv('data/passhe_data/Total_Headcount.csv', encoding='utf-16', sep='\t')
        totalHeadcount.name = "TotalHeadcount"

        fullTime = pd.read_csv('data/passhe_data/FullTime.csv', encoding='utf-16', sep='\t')
        fullTime.name = "FullTime"

        livingOnCampus = pd.read_csv('data/passhe_data/LivingOnCampus.csv', encoding='utf-16', sep='\t')
        livingOnCampus.name = "LivingOnCampus"

        adultLearner = pd.read_csv('data/passhe_data/AdultLearner.csv', encoding='utf-16', sep='\t')
        adultLearner.name = "AdultLearner"

        educationMajor = pd.read_csv('data/passhe_data/EducationMajor.csv', encoding='utf-16', sep='\t')
        educationMajor.name = "EducationMajor"

        minority = pd.read_csv('data/passhe_data/Minority.csv', encoding='utf-16', sep='\t')
        minority.name = "Minority"

        stem_hMajor = pd.read_csv('data/passhe_data/STEM-H_Major.csv', encoding='utf-16', sep='\t')
        stem_hMajor.name = "STEMHMajor"

        transferStudent = pd.read_csv('data/passhe_data/TransferStudent.csv', encoding='utf-16', sep='\t')
        transferStudent.name = "TransferStudent"
        """

        data = [higherEdBySector,
                higherEdInstitutions,
                cost]
        return data

    def merge(self):
        data = self.collect()
        for frame in data:
            new_cols = list(frame.columns)
            for i in range(len(new_cols)):
                new_cols[i] = new_cols[i] + f" ({frame.name})"
            #print(new_cols)
            frame.rename(columns=dict(zip(frame.columns, new_cols)), inplace=True)
        # Column merging - insitution name
        data[0] = data[0].rename(columns={"College (HigherEducationBySector)" : "Institution"})
        data[1] = data[1].rename(columns={"Institution Name (HigherEducationInstitutions)" : "Institution"})
        data[2] = data[2].rename(columns={"Institution Name (Cost)" : "Institution"})

        data[0] = data[0].rename(columns={"County (HigherEducationBySector)": "County"})
        data[1] = data[1].rename(columns={"County (HigherEducationInstitutions)": "County"})

        data[0] = data[0].rename(columns={"Sector (HigherEducationBySector)": "Category"})
        data[1] = data[1].rename(columns={"Category (HigherEducationInstitutions)": "Category"})

        # Column dropping
        data[0].drop(columns=["Georeferenced Latitude & Longitude (HigherEducationBySector)", "Sector and Type (HigherEducationBySector)"], inplace=True)
        data[1].drop(columns=["Administrative Unit Number (HigherEducationInstitutions)",
                              "School Branch (HigherEducationInstitutions)",
                              "Status (HigherEducationInstitutions)",
                              "Phone Number (HigherEducationInstitutions)",
                              "Phone_Number_Extn (HigherEducationInstitutions)",
                              "Web_Address (HigherEducationInstitutions)",
                              "Primary Administrator (HigherEducationInstitutions)",
                              "School Year (HigherEducationInstitutions)"], inplace=True)
        data[2].drop(columns=["UnitID (Cost)",
                              "Unnamed: 9 (Cost)"], inplace=True)

        # Merge
        dataset = pd.concat(data, axis=0)
        #print(dataset.head())
        #print(dataset.shape)
        dataset.to_csv("data.csv", index=False)
        return dataset

    def clean_encode_split(self):
        dataset = self.merge()
        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Other")
        preprocessor = ColumnTransformer([
            ("num_imputer", num_imputer, ["zip code (HigherEducationBySector)", "Latitude (HigherEducationBySector)", "Longitude (HigherEducationBySector)", "Total price for in-district students living on campus  2024-25 (DRVCOST2024) (Cost)", "Total price for in-state students living on campus 2024-25 (DRVCOST2024) (Cost)", "Total price for in-district students living off campus (not with family)  2024-25 (DRVCOST2024) (Cost)", "Total price for in-state students living off campus (not with family)  2024-25 (DRVCOST2024) (Cost)", "Total price for in-district students living off campus (with family)  2024-25 (DRVCOST2024) (Cost)", "Total price for in-state students living off campus (with family)  2024-25 (DRVCOST2024) (Cost)"]),
            ("cat_imputer", cat_imputer, ["Category", "Type (HigherEducationBySector)", "County", "Address (HigherEducationBySector)", "City (HigherEducationInstitutions)"])
        ])
        dataset["State abbreviation (HD2024) (Cost)"] = dataset["State abbreviation (HD2024) (Cost)"].fillna("PA")

        le = LabelEncoder()
        cols = ["Category", "Type (HigherEducationBySector)", "County", "Address (HigherEducationBySector)", "City (HigherEducationInstitutions)"]
        for col in cols:
            dataset[col] = le.fit_transform(dataset[col])

        x = dataset.drop(columns=['Institution'], inplace=False)
        y = dataset['Institution']
        x = preprocessor.fit_transform(x.astype(object)) # Google search AI Overview
        print(y)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        return train_x, test_x, train_y, test_y

    def scale(self):
        train_x, test_x, train_y, test_y = self.clean_encode_split()
        ss = StandardScaler()
        train_x = ss.fit_transform(train_x)
        test_x = ss.transform(test_x)

        return train_x, test_x, train_y, test_y

    #
    def nanCheck(self):
        data = self.merge()
        for col in data:
            print(f"Total NaN in {col}: {data[col].isna().sum()}")



#data = Data()
#data.collect()
#data.merge()
#data.clean_split()
#data = data.encode_scale()
#data.nanCheck()
