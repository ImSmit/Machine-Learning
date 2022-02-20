import pandas as pd
ds = pd.read_csv("./Dataset/BreastCancerDataset.csv")

print(ds.to_string())
X = ds[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']]
Y = ds['diagnosis']

import sklearn as sk
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=42)
#MLPClassifier Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs',alpha=0.00001,hidden_layer_sizes=(5,),max_iter=23)

classifier.fit(x_train,y_train)
test_data = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print("accuracy_score",accuracy_score(y_test,test_data))
