# Developement and train of multiple classfiication models into pipelines and merge into ensemble model

from dataPre import Data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class Models:
    def __init__(self):
        d = Data()
        data = d.split()

    def KNN(self):
        model = KNeighborsClassifier()
        return model

    def DTC(self):
        model = DecisionTreeClassifier()
        return model

    def RFC(self):
        model = RandomForestClassifier()
        return model

    def LRC(self):
        model = LogisticRegression()
        return model

    def gen(self):
        # Can change which models we add into the ensemble depending on the performance and data
        KNN = self.KNN()
        DTC = self.DTC()
        RFC = self.RFC()
        LRC = self.LRC()
        #ensembleModel = Pipeline()
        #return ensembleModel
