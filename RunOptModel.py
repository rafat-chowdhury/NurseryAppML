import pandas as pd 
import numpy as np 
import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier

# Define some basic functions
def readInAndCheckData(filename):
    data = pd.read_csv(filename)
    print(data.head())
    print(data.describe())
    return data

# Read in file and retrieve summary stats
filename = "nursery.data"
data = readInAndCheckData(filename)

# Read in optimised model
NurseryAppMLMod_in = open("NurseryAppML.pickle","rb") # Load in model with best accuracy
NurseryAppMLMod = pickle.load(NurseryAppMLMod_in) 

# labels = [range(1,9)]
l1 = int(input('Parental suitability (rate: 1-3):'))
l2 = int(input('Nursery attended (rate: 1-5):'))
l3 = int(input('Form completion (rate: 1-4):'))
l4 = int(input('No. of children:'))
l5 = int(input('Housing situation (rate: 1-3):'))
l6 = int(input('Financial situation (rate: 1-2):'))
l7 = int(input('Social standing (rate: 1-3):'))
l8 = int(input("Child's health (rate: 1-3):"))

labels = ([l1,l2,l3,l4,l5,l6,l7,l8])
labels = np.array(labels)
labels = labels.reshape(1, -1)
predicted = NurseryAppMLMod.predict(labels)
predicted = int(predicted)

# Define sublabels for original data
names = ["very_recom","recommended", "priority", "not_recom"]
print("Predicted application output:", names[predicted])