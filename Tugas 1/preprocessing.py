import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



dataset = pd.read_csv('dataAkta.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#print(x)
#print(y)




imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

maks = pd.options.display.max_rows

imputer.fit(X[:, 1:3])




X[:, 1:3] = imputer.transform(X[:, 1:3])


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

#print(X)

le = LabelEncoder()
y = le.fit_transform(y)

#print(X_train)

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_test)