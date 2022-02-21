import pandas as pd

ds = pd.read_csv("./Dataset/income.csv")
print(ds.info())
x = ds[['Age']]
y = ds[['Income']]

import sklearn as sk
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=30)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
print("coeffisient",model.coef_)
print("Intercept",model.intercept_)
print("Rank",model.rank_)

mpredict = model.predict(x_test)
#meansquare
from sklearn.metrics import r2_score
import numpy as np
np.mean(np.absolute(y_test - mpredict))
meansquare = np.mean((y_test - mpredict) ** 2)
print("Meansquare {0}".format(meansquare))

r2score = r2_score(mpredict,y_test)
print("r2Score is : {0}".format(r2score))
