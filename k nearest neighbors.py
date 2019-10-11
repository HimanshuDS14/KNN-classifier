import numpy as np
import pandas as pd

from sklearn import preprocessing  , model_selection , neighbors

data = pd.read_csv("breast-cancer-wisconsin.data")
print(data.head(10))

#handle missing data

data.replace('?' , -99999 , inplace=True)

#drop useless data
data.drop(['id'] , axis=1 , inplace=True)

print(data.head(10))

x = np.array(data.drop(["Class"] , axis=1)) #feature is everything except class column
y = np.array(data["Class"])    #label is class column


train_x , test_x , train_y , test_y = model_selection.train_test_split(x,y , test_size=0.2)

classifier=  neighbors.KNeighborsClassifier()
classifier.fit(train_x , train_y)


accuracy = classifier.score(test_x , test_y)
print(accuracy)

example_measure = np.array([4,8,7,10,4,10,7,5,1])
example_measure = example_measure.reshape(1,-1)

prediction = classifier.predict(example_measure)
print(prediction)