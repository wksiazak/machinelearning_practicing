import seaborn
import pandas as pd
from matplotlib import pyplot
from numpy import nan
from pandas import get_dummies
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Classifiers:
    def splitDatasetIntoTrainAndTest(self, X, y, train_split_percent = 0.6):
        # pd.set_option('display.max_columns', None)
        # print(X)
        print(X.info())
        # print(X.describe())
        # print(X.describe(include=[pd.np.number]))
        # print(X.describe(include=[pd.np.object]))
        # print(X.describe(include=['category']))
        # print(X.describe(include={'boolean'}))
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_split_percent)
        return X_train, X_test, y_train, y_test
    def datasetPreprocessing(self, X, columns_to_drop, columns_to_map,
                             nan_to_median_columns, nan_to_most_freq_columns):
        # usuwanie
        X_clean = X.drop(columns_to_drop, axis=1)
        # mapowanie
        for column_name in columns_to_map:
            # konstruowanie mappera
            mapper = {}
            for index, category in enumerate(X_clean[column_name].unique()):
                mapper[category] = index
            # mapowanie
            X_clean[column_name] = X_clean[column_name].map(mapper)
        # uzupełnianie
        for column_name in nan_to_median_columns:
            X_clean[column_name] = X_clean[column_name].fillna(X_clean[column_name].median())
        # for column_name in nan_to_most_freq_columns:
        #     impFreq = SimpleImputer(missing_values=nan, strategy='most_frequent')
        #     X_clean[column_name] = impFreq.fit_transform(X_clean[column_name])
        return X_clean
    def trainAndTestClassifier(self, clf, X_train, X_test, y_train):
        print(clf)
        # trenowanie
        clf.fit(X_train, y_train)
        # testowanie
        y_pred = clf.predict(X_test)
        return y_pred
    def getClassificationScore(self, clf_name ,y_test, y_pred):
        print("Nazwa klasyfikatora: " + clf_name)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
    def crossValidation(self, clf, clf_name, X, y, folds=5):
        print("cross-validation: " + clf_name)
        scores = cross_val_score(clf, X, y, cv = folds)
        print(scores)
        # wynik mean i stddev testów kroswalidacji
        print("mean: " + str(scores.mean()))
        print("stddev: " + str(scores.std()))
    def ensableClassifier(self, clfs, X_train, X_test, y_train):
        y_preds = []
        # trenowanie i testowanie wszystkich klasyfikatorów z listy clfs
        for clf in clfs:
            clf.fit(X_train, y_train)
            y_preds.append(clf.predict(X_test))
        # głosowanie większościowe
        y_result = y_preds[0]
        clf_index = 1
        while(clf_index < len(y_preds)):
            index = 0
            while(index < len(y_result)):
                y_result[index] = y_result[index] + y_preds[clf_index][index]
                index += 1
            clf_index += 1
        # uśrednianie i zaokrąglanie
        for index, y in enumerate(y_result):
            y_result[index] = round(y_result[index]/len(clfs))
        return y_result
    def plotClassificationResult(self, column1, x_label, column2, y_label, y_pred):
        pyplot.scatter(column1, column2, c=y_pred)
        pyplot.title("Klasyfikacja próbek")
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        pyplot.show()

c = Classifiers()
# X_clean = c.datasetPreprocessing(
#     X = seaborn.load_dataset("titanic").iloc[:, 1:],
#     columns_to_drop = ['sex','embarked','class','adult_male','deck','alive'],
#     columns_to_map = ['who','embark_town', 'alone'],
#     nan_to_median_columns = ['age'],
#     nan_to_most_freq_columns = []
#     )
# X_train, X_test, y_train, y_test = c.splitDatasetIntoTrainAndTest(
#     X=X_clean,
#     y=seaborn.load_dataset("titanic")['survived'],
#     train_split_percent = 0.6
# )
# y_pred_knn5_train = c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_train,y_train)
# y_pred_knn5_test = c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_test,y_train)
# y_pred_tree_train = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_train,y_train)
# y_pred_tree_test = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_test,y_train)
# y_pred_svm_lin_train = c.trainAndTestClassifier(SVC(kernel='linear'), X_train,X_train,y_train)
# y_pred_svm_lin_test = c.trainAndTestClassifier(SVC(kernel='linear'), X_train,X_test,y_train)
# y_pred_svm_rbf_train = c.trainAndTestClassifier(SVC(kernel='rbf', gamma='auto'), X_train,X_train,y_train)
# y_pred_svm_rbf_test = c.trainAndTestClassifier(SVC(kernel='rbf', gamma='auto'), X_train,X_test,y_train)
#
# c.getClassificationScore("kNN-5 trenowanie", y_train, y_pred_knn5_train)
# c.getClassificationScore("kNN-5 testowanie", y_test, y_pred_knn5_test)
# c.getClassificationScore("DT trenowanie", y_train, y_pred_tree_train)
# c.getClassificationScore("DT testowanie", y_test, y_pred_tree_test)
# c.getClassificationScore("SVM-linear trenowanie", y_train, y_pred_svm_lin_train)
# c.getClassificationScore("SVM-linear testowanie", y_test, y_pred_svm_lin_test)
# c.getClassificationScore("SVM-rbf trenowanie", y_train, y_pred_svm_rbf_train)
# c.getClassificationScore("SVM-rbf testowanie", y_test, y_pred_svm_rbf_test)

# cross-validation
# czyszczenie zbioru
X_clean = c.datasetPreprocessing(
    X = seaborn.load_dataset("titanic").iloc[:, 1:],
    columns_to_drop = ['sex','embarked','class','adult_male','deck','alive'],
    columns_to_map = ['who','embark_town', 'alone'],
    nan_to_median_columns = ['age'],
    nan_to_most_freq_columns = []
    )
# strojenie algorytmu
clf = RandomForestClassifier()
c.crossValidation(clf, 'RF', X_clean, seaborn.load_dataset("titanic")['survived'], folds=5)
# podział na train i test
X_train, X_test, y_train, y_test = c.splitDatasetIntoTrainAndTest(
    X=X_clean,
    y=seaborn.load_dataset("titanic")['survived'],
    train_split_percent = 0.6
)
# trenowanie i testowanie
y_pred_svm_lin_train = c.trainAndTestClassifier(clf, X_train,X_train,y_train)
y_pred_svm_lin_test = c.trainAndTestClassifier(clf, X_train,X_test,y_train)
# wyniki
c.getClassificationScore("RF trenowanie", y_train, y_pred_svm_lin_train)
c.getClassificationScore("RF testowanie", y_test, y_pred_svm_lin_test)

# klasyfikacja zespołowa
y_pred_ensable_train = c.ensableClassifier(
    [SVC(kernel='linear'), SVC(), KNeighborsClassifier()], X_train, X_train, y_train)
y_pred_ensable_test = c.ensableClassifier(
    [SVC(kernel='linear'), SVC(), KNeighborsClassifier()], X_train, X_test, y_train)
c.getClassificationScore("Uczenie zespołowe trenowanie", y_train, y_pred_ensable_train)
c.getClassificationScore("Uczenie zespołowe testowanie", y_test, y_pred_ensable_test)
c.plotClassificationResult(X_test['age'],'age', X_test['fare'], 'fare', y_pred_ensable_test)