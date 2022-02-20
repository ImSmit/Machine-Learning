import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

da = pd.read_csv('./Dataset/BreastCancerDataset.csv')

print(da.info())
print(da.describe())


x=da[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
#dependent data
y=da[['diagnosis']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

c = MLPClassifier(solver='lbfgs',alpha=0.00001,hidden_layer_sizes=(5,),max_iter=20)
c.fit(x_train,y_train)
test_data=c.predict(x_test)
##trait_data=c.predict(x_train)

print("Accuracy Score: ",accuracy_score(y_test,test_data))
print("Confusion Matrix: \n",confusion_matrix(y_test,test_data))

