import pandas as pd
import seaborn
from numpy import nan
from pandas import DataFrame
from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class MLWarmup:

    def getPlanetsDataset(self):
        planets_df = load_dataset("planets")
        print(planets_df)
        column_names = ['method', 'number', 'orbital_period', 'mass', 'distance', 'year']
        # ilość wartości pustych w kolumnach
        print(planets_df.isnull().sum())
        # usuń wszystkie kolumny zawierające więcej niż połowę wartości pustych - 1035
        half_dataset_no = int(len(planets_df)/2)
        planets_df1 = planets_df
        for c in column_names:
            if(planets_df[c].isnull().sum() > half_dataset_no):
                planets_df1 = planets_df1.drop(c,axis=1)
        print(planets_df1.isnull().sum())
        # usuń wszystkie te wiesze kóre zawierają ponad połowę wartości pustych -> planets_df
        planets_df2 = planets_df
        planets_df2 = planets_df2.dropna(thresh=4)
        print(planets_df2)
        # uzupełnianie pustych danych
        impFreq = SimpleImputer(missing_values=nan, strategy='most_frequent')
        planets_df3 = impFreq.fit_transform(planets_df)
        planets_df3 = DataFrame(planets_df3, columns=list(planets_df.columns))
        print(planets_df3.isnull().sum())
        print(planets_df3)
        # mapowanie danych jakościowych na liczby porządkowe
        planets_df4 = planets_df
        method_mapper = {}
        for index, num_category in enumerate(planets_df.method.unique()):
            method_mapper[num_category] = index
        print(method_mapper)
        planets_df4['method'] = planets_df['method'].map(method_mapper)
        # print(planets_df4)
        # uzupełnienie wartości pustych
        impMean = SimpleImputer(missing_values=nan, strategy='mean')
        planets_df4 = impMean.fit_transform(planets_df)
        planets_df4 = DataFrame(planets_df4, columns=list(planets_df.columns))
        print(planets_df4.isnull().sum())
        print(planets_df4)
        # skalowanie danych
        std = StandardScaler()
        planets_df4 = std.fit_transform(planets_df4)
        planets_df4 = DataFrame(planets_df4, columns=list(planets_df.columns))
        print(planets_df4)
        corr = planets_df4.corr()
        pd.set_option('display.max.columns', None)
        print(corr)

    def getIrisDataset(self):
        self.iris = load_iris()
        index = 0
        while(index < len(self.iris['target'])):
            print(self.iris['data'][index], self.iris['target'][index], self.iris['target_names'][self.iris['target'][index]],
                  sep=' | ')
            index += 1
    def splitDataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.iris['data'], self.iris['target'], train_size=0.2)
        index = 0
        print("TRENINGOWY", len(self.X_train))
        while (index < len(self.y_train)):
            print(self.X_train[index], self.y_train[index],sep=' | ')
            index += 1
    def trainModel(self):
        self.knn3 = KNeighborsClassifier(n_neighbors=3, metric='chebyshev')
        self.knn5 = KNeighborsClassifier(n_neighbors=5)
        self.knn7 = KNeighborsClassifier(n_neighbors=7)
        # trenowanie
        self.knn3.fit(self.X_train,self.y_train)
        self.knn5.fit(self.X_train,self.y_train)
        self.knn7.fit(self.X_train,self.y_train)
        # pomiar dokładności trenowania !!!
        y_pred3 = self.knn3.predict(self.X_train)
        y_pred5 = self.knn5.predict(self.X_train)
        y_pred7 = self.knn7.predict(self.X_train)
        print("ACC(kNN3) train: " + str(accuracy_score(self.y_train, y_pred3)))
        print("ACC(kNN5) train: " + str(accuracy_score(self.y_train, y_pred5)))
        print("ACC(kNN7) train: " + str(accuracy_score(self.y_train, y_pred7)))
    def testModel(self):
        y_pred3 = self.knn3.predict(self.X_test)
        y_pred5 = self.knn5.predict(self.X_test)
        y_pred7 = self.knn7.predict(self.X_test)
        print("ACC(kNN3) test: " + str(accuracy_score(self.y_test, y_pred3)))
        print("ACC(kNN5) test: " + str(accuracy_score(self.y_test, y_pred5)))
        print("ACC(kNN7) test: " + str(accuracy_score(self.y_test, y_pred7)))
        print(confusion_matrix(self.y_test, y_pred3))
    def homework(self):
        self.titanic = seaborn.load_dataset("titanic")
        print(self.titanic)
        # y -> survived
        # X -> reszta kolumn
        # 1. Oczyszczenie danych i podział na zbiór testowy i treningowy ???
        # 2. Trenowanie na podstawie kNN(???)
        # 3. Testowanie
        # 4. Ocena klasyfikacji i testowania
ml = MLWarmup()
ml.getIrisDataset()
ml.splitDataset()
ml.trainModel()
ml.testModel()
ml.homework()
# ml.getPlanetsDataset()