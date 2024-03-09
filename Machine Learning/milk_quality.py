from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter("ignore")


milk_data = pd.read_csv('milknew.csv')

"""
print(milk_data)

print(milk_data['Grade'].value_counts())
plt.figure(figsize=(12,8))
plt.title("Grades of Milk",fontsize=15)
c1=sns.countplot(x='Grade',data=milk_data)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,8))
plt.title("pH of Milk",fontsize=15)
c1=sns.countplot(x='pH',data=milk_data)
plt.xticks(rotation=45)
plt.show()

data = milk_data.rename(columns={'Temprature':'temperature'})
plt.figure(figsize=(10,8))
plt.title("Temperature of Milk",fontsize=15)
c1=sns.countplot(x='temperature',data=data)
# c1.bar_label(c1.containers[0],size=12)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,8))
plt.title("Colour of Milk",fontsize=15)
c1=sns.countplot(x='Colour',data=milk_data)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(milk_data.corr(),annot=True)
plt.show()
"""

x=milk_data.iloc[:,:7]
y=milk_data.iloc[:,7]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=7)


model=LogisticRegression()
model.fit(x_train,y_train)
y_prediction=model.predict(x_test)
print(confusion_matrix(y_test,y_prediction))
accuracy=accuracy_score(y_test,y_prediction)
print("Accuracy of the logistic regression model is {}".format(accuracy))


model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)
y_prediction=model2.predict(x_test)
print(confusion_matrix(y_test,y_prediction))
accuracy=accuracy_score(y_test,y_prediction)
print("Accuracy of the decision tree classifier model is {}".format(accuracy))

model3=KNeighborsClassifier()
model3.fit(x_train,y_train)
y_prediction=model3.predict(x_test)
print(confusion_matrix(y_test,y_prediction))
accuracy=accuracy_score(y_test,y_prediction)
print("Accuracy of the K Neighbors Classifier model is {}".format(accuracy))


input = (6.7,38,1,0,1,0,255)
inputs_grade = ["high"]
print("The given inputs milk quality is: {}".format(inputs_grade))
nparray = np.asarray(input)

reshape = nparray.reshape(1,-1)

prediction = model.predict(reshape)
print("Prediction for Logistic Regression Model: {}".format(prediction))

prediction2 = model2.predict(reshape)
print("Prediction for Decision Tree Model: {}".format(prediction2))

prediction3 = model3.predict(reshape)
print("Prediction for K Neighbors Model: {}".format(prediction3))
