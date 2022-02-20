import pandas as pd

ds = pd.read_csv("./Dataset/diabetes.csv")

X = ds[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

Y = ds['Outcome']

#train
import sklearn as sk
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 50)

#built model
from sklearn import neighbors
model = sk.neighbors.KNeighborsClassifier()
model.fit(X_train,Y_train)#train data nakhvana

#data predict karavana
print("predict")
dataclass = model.predict(X_test)
##
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,dataclass,labels=[0,1]))


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,dataclass))



        
