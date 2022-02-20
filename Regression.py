import pandas as pd

ds = pd.read_csv("./Dataset/income.csv")
print(ds.to_string())

import matplotlib.pyplot as plt
plt.scatter(ds.Age,ds.Income,color='green')#to make point on a plot
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

##import sklearn.models
from sklearn.model_selection import train_test_split
