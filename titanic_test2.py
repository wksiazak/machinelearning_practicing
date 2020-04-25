import pandas as pd
from numpy import nan
from pandas import DataFrame
from seaborn import load_dataset
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

titanic = load_dataset('titanic')
df_titanic = pd.DataFrame(titanic)
print(df_titanic)
print(df_titanic.shape)
print(df_titanic.isnull().sum())
print(df_titanic.dtypes)

df_titanic.dropna( axis=1)


print(df_titanic.isnull().sum())


X = df_titanic['sibsp']
y=df_titanic['pclass']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
knn4 = KNeighborsClassifier(n_neighbors=4)
knn4.fit(X_train, y_train)
y_pred4 = knn4.predict(X_train)
print("ACC(kNN4) train: " + str(accuracy_score(y_train, y_pred4)))
y_pred3 = knn4.predict(X_test)
print("ACC(kNN4) test: " + str(accuracy_score(y_test, y_pred4)))