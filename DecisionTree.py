#DesicionTree
import pandas as pd

ds = pd.read_csv("./Dataset/Iris Dataset.csv")
print(ds.to_string())
x = ds[['Sepal Length','Sepal Width','Petal Length','Petal Width']]
y = ds[['Species']]

import sklearn as sk
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=30)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(x_train,y_train)
mpredict = model.predict(x_test)

print(model.score(x_test,y_test))
