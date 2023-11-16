import numpy as np
import pandas as pd

train_data = pd.read_csv('train.csv')
train_data.head()

test_data = pd.read_csv('test.csv')
test_data.head()



X = train_data.drop('label', axis=1)
y = train_data['label']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model = KNeighborsClassifier()
model.fit(X_train, y_train)

predict = model.predict(X_test)
print('knn', accuracy_score(predict, y_test))




clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('knn', accuracy_score(pred, y_test))


#output for submit

true_pred = clf.predict(test_data)
samp = pd.read_csv('sample_submission.csv')
samp['Label'] = true_pred
samp.to_csv('sub.csv', index=False)