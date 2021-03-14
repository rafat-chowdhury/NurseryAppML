import pandas as pd 
import numpy as np 
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

def readInAndCheckData(filename):
    data = pd.read_csv(filename)
    print(data.head())
    print(data.describe())
    return data

def sortdata(labels,orig_data,testvol):
    label_train, label_test, orig_data_train, orig_data_test = sklearn.model_selection.train_test_split(labels,orig_data,test_size = testvol)
    return label_train, label_test, orig_data_train, orig_data_test

def trainKNNmodel(label_train,orig_data_train,label_test,orig_data_test,k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(label_train,orig_data_train)
    acc = model.score(label_test,orig_data_test)
    print(acc)
    return model,acc

# Read in file and retrieve summary stats
filename = "nursery.data"
data = readInAndCheckData(filename)

# Preprocess table headers 
labenc = preprocessing.LabelEncoder() # Format string data into numerical using label encoder
parents = labenc.fit_transform(list(data["parents"]))
has_nurs = labenc.fit_transform(list(data["has_nurs"]))
form = labenc.fit_transform(list(data["form"]))
children = labenc.fit_transform(list(data["children"]))
housing = labenc.fit_transform(list(data["housing"]))
finance = labenc.fit_transform(list(data["finance"]))
social = labenc.fit_transform(list(data["social"]))
health = labenc.fit_transform(list(data["health"]))
assessment = labenc.fit_transform(list(data["assessment"]))

# Identify labels and original data to be predicted
labels = list(zip(parents,has_nurs,form,children,housing,finance,social,health))
orig_data = list(assessment)

# Separate labels and original data into test and train data
testvol = 0.9 # Vary this < value gives higher acc
label_train, label_test, orig_data_train, orig_data_test = sortdata(labels,orig_data,testvol)

# Train model and pass through test data
k = 7 # Need to optimise this
model,acc = trainKNNmodel(label_train,orig_data_train,label_test,orig_data_test,k)
predicted = model.predict(label_test)

# Define sublabels for original data
names = ["very_recom","recommended", "priority", "not_recom"]

# Visualise prediction accuracy
for x in range(len(orig_data_test)):
    print("Predicted:", names[predicted[x]], "Actual: ", names[orig_data_test[x]])