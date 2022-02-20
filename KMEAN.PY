import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##ds = pd.read_csv("./Dataset/diabetes.csv")
##
##print(ds.iloc[:,[0,4]].values)
X = np.array([[1, 2],  
              [1.5, 1.8],  
              [5, 8 ],  
              [8, 8],  
              [1, 0.6],  
              [9,11],  
              [1,3],  
              [8,9],  
              [0,3],  
              [5,4],  
              [6,4],  
              [1,2],  
              [2,4],  
              [11,5],  
              [2.56,5.6],  
              [2, 2],  
              [1.8, 1.8],  
              [8, 8 ],  
              [5, 5],  
              [0.6, 0.6],  
              [11,11],  
              [3,3],  
              [9,9],])

class KMeans:
    def __init__(self,K=5,max_iter=100):
        self.k = K
        self.max_iter = max_iter
    def fit(self,data):
        for i in rannge(self.k):
            self.centroid[i] = data[i]

        for i in range(self.max_iter):
            self.classfications = {}

        for feature in data:
            

























