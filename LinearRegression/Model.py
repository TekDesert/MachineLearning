import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

''' 

Reading the CSV and selecting the Data (x and y) for our linear regression model

'''

data = pd.read_csv("student-mat.csv", sep=";")

print( data.head())

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3" #we would like to predit the G3 (third semester grade) of a student


x = np.array(data.drop(predict, 1)) #we will use everything exept G3 for our x value
y = np.array(data[predict]) #we will only use G3 for our y value
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) 

''' 

Creation of our testing set and linear model, we'll train it 30 times and save the best occurance


best = 0

for loop in range (30): 

    #Select a part of the data (x_train and y_train) that we will use to train the algorithm. (x_test, y_test) will be used to verify those results
    #we are splitting here 10% of our data into test samples (x_test, y_test) to see if our computer can guess it
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1) 

    #No Need to retrain our model now that we have saved it in pickle

    linear = linear_model.LinearRegression()

    #it will fit the data to find a best fit lign
    linear.fit(x_train, y_train)
    modelAccuracyScore = linear.score(x_test, y_test)


    #Our Model is now created

    print("The model was able to guess " + str(round((modelAccuracyScore)*100)) + "% of the values correctly \n")

    #linear.coef_ is actually the 'a' coefficient in our line (y=ax+b)
    print("a-Coefficient: \n", linear.coef_)

    #linear.intercept_ is actually the 'y=0' value of x our the line (y=ax+b)
    print("Y-Intercept: \n", linear.intercept_, "\n")


    #Exporting the model with pickle and reimporting it

    if modelAccuracyScore > best: #if our current accuracy score for our model this loop is the best we ever had, we'll save it
        best = modelAccuracyScore
        with open("studentmodel.pickle","wb") as f:  #we are saving our model in a '.pickle' file
            pickle.dump(linear,f) #we are dumping our model 'linear' into the file 'f'"""

End of Model Training
'''

pickle_in = open("studentmodel.pickle", "rb") #opening back our file

linear = pickle.load(pickle_in) #We are loading back our model into a variable

#print("\n Our final best model had an accuracy of " + str(round((best)*100)) + "%") 

''' 

let's see our predicted values



predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    #predictions[x] is the grade the student was predicted to earn
    #x_test[x] is the student's date (in order : G1, G2, Study time, failures, absence)
    #y_test[x] is the student's actual grade

'''

''' 

Information display in the form of a graph | we will display here the corrolation of a feature "p" with the final grade of a student

'''

p = "absences" #Pick one of the features 

style.use("ggplot")

pyplot.scatter(data[p], data["G3"]) #Using a scatter plot with x and y

pyplot.xlabel(p) #x label name
pyplot.ylabel("Final Grade") #y label name

pyplot.show()
