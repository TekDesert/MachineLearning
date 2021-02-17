import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing


''' Opening and reading our file with pandas + Formating the non Integer Data '''

data = pd.read_csv("../car.data")
print(data.head())

#We have to convert non integer values into integers (example "medium" -> 1 | "hard" -> 2)
le = preprocessing.LabelEncoder() 

#We will take for example here the entire "buying" column and turn it into a list, then transform them into integers values
buying = le.fit_transform(list(data["buying"])) 
maint = le.fit_transform(list(data["maint"])) 
door = le.fit_transform(list(data["door"])) 
persons = le.fit_transform(list(data["persons"])) 
lug_boot = le.fit_transform(list(data["lug_boot"])) 
safety = le.fit_transform(list(data["safety"])) 
clas = le.fit_transform(list(data["class"])) 

print(buying)

''' Setting up the Model '''

predict = "class" #what we want to predict

x = list(zip(buying, maint, door, persons, lug_boot, safety)) #Our features
y = list(clas) #Our labels

 #once you go over 0.2 you sacrifice your data and performance so it isn't recommended
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

print(x_train, y_test)