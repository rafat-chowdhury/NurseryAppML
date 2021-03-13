import pandas as pd 
import numpy as np 
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("nursery.data")
print(data.head())

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

predict = "class"

labels = list(zip(parents,has_nurs,form,children,housing,finance,social,health))
orig_data = list(assessment)

testvol = 0.9
label_train, label_test, orig_data_train, orig_data_test = sklearn.model_selection.train_test_split(labels,orig_data,test_size = testvol)

k = 7
model = KNeighborsClassifier(n_neighbors=k)

model.fit(label_train,orig_data_train)
acc = model.score(label_test,orig_data_test)
print(acc)

predicted = model.predict(label_test)
names = ["very_recom","recommended", "priority", "not_recom"]

for x in range(len(orig_data_test)):
    print("Predicted:", names[predicted[x]], "Actual: ", names[orig_data_test[x]])