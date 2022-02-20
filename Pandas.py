import pandas as pd

ds = pd.read_csv('./Dataset/diabetes.csv')
#dataset no info aape
print("----------------dataset no info aape")
print("----------------ds.info()")
print(ds.info())

#0 thi 5 value print kare
print("----------------0 ne 5 value print kare")
print("----------------ds.loc[[0,5]]")
print(ds.loc[[0,5]])

###aakha data set ne print karava mate
##print("----------------aakha data set ne print karava mate")
##print("----------------ds.to_string()")
##print(ds.to_string())
##
#iloc
# 3rd column ma 8th row ni value value
print("----------------8th row ma 3rd colums ni value value")
print("----------------ds.iloc[8,3]")
print(ds.iloc[8,3])

#top 5 row
print("----------------ds.head(5)")
print(ds.head(5))

#last 11 row
print("----------------ds.tail(11)")
print(ds.tail(11))
