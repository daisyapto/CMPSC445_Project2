# Developement and train of multiple classfiication models into pipelines and merge into ensemble model

from dataPre import Data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

class Models:
    def __init__(self):
        self.d = Data()
        self.train_x, self.test_x, self.train_y, self.test_y = self.d.scale()

    def KNN(self):
        model = KNeighborsClassifier()
        pipe = Pipeline([('knn', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def DTC(self):
        model = DecisionTreeClassifier()
        pipe = Pipeline([('dt', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def RFC(self):
        model = RandomForestClassifier()
        pipe = Pipeline([('rf', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def LRC(self):
        model = LogisticRegression()
        pipe = Pipeline([('lr', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def gen(self):
        # Can change which models we add into the ensemble depending on the performance and data
        KNN, score1 = self.KNN()
        DTC, score2 = self.DTC()
        RFC, score3 = self.RFC()
        LRC, score4 = self.LRC()
        print("KNN Score: ", score1)
        print("DTC Score: ", score2)
        print("RFC Score: ", score3)
        print("LRC Score: ", score4)
        ensembleModel = VotingClassifier([('KNN', KNN), ('DTC', DTC), ('RFC', RFC), ('LRC', LRC)], voting='soft')
        return ensembleModel

    def train(self, model):
        model = self.gen()
        print(self.train_y.shape)
        model.fit(self.train_x, self.train_y)
        return model.score(self.train_x, self.train_y), model.score(self.test_x, self.test_y)
        # (0.5983791085096803, 0.3237410071942446) --> needs improvement

models = Models()
models.gen()