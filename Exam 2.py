#ANN
import pandas as pd

ds = pd.read_csv("./Dataset/BreastCancerDataset.csv")

##print(ds.info())
##print(ds.to_string())
x = ds[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean']]
ds['diagnosis'] = ds['diagnosis'].map({'M':1,'B':0})
print(ds['diagnosis'])
y = ds['diagnosis']
import sklearn as sk
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=60)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs',alpha = 0.00001,hidden_layer_sizes = (5,),max_iter=23)

model.fit(x_train,y_train)

mpredict = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,mpredict))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,mpredict,labels=[0,1]))
