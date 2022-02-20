import pandas as pd

ds = pd.read_csv("./Dataset/income.csv")
print(ds.to_string())


import matplotlib.pyplot as plt
plt.scatter(ds.Age,ds.Income,color='green')#to make point on a plot
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

x = ds[['Age']]
y = ds[['Income']]
##import sklearn.models
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.33,random_state=30)

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

print("Coefficient",model.coef_)
print("Intercept",model.intercept_)
print("Rank",model.rank_)


from sklearn.metrics import r2_score
import numpy as np
mpredict = model.predict(x_test)
np.mean(np.absolute(mpredict - y_test))
meansquare = np.mean((mpredict - y_test) ** 2)
print('Mean square error is {0} : '.format(meansquare))

r2score = r2_score(mpredict,y_test)
print('score = {0}'.format(r2score))


















