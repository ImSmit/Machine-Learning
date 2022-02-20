import pandas as pd

ds = pd.read_csv("./Dataset/avocado.csv")

##print(ds.to_string())

X = ds[['AveragePrice','Total Volume','Total Bags']]
Y = ds['type']

import sklearn as sk
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(X,Y,test_size = 0.33,random_state=30)

from sklearn import svm
from sklearn.svm import LinearSVC
model = svm.SVC()
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
spredict = model.predict(x_test)
print("Accuracy score",accuracy_score(y_test,spredict))

