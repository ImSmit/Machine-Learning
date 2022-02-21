import pandas as pd

ds = pd.read_csv("./Dataset/avocado.csv")

##print(ds.to_string())
x = ds[['AveragePrice','Total Volume','Total Bags','Small Bags','Large Bags']]
ds['type'] = ds['type'].map({'conventional':1,'organic':0})
print(ds['type'])
import sklearn as sk
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=40)

from sklearn import neighbors
model = sk.neighbors.KNeighborsClassifier()

model.fit(x_train,y_train)
mpredict = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,mpredict))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,mpredict,labels=[0,1]))
