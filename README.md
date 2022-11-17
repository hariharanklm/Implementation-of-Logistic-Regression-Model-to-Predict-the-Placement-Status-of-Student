# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Hariharan M
RegisterNumber:  212221220015
*/
```
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![1](https://user-images.githubusercontent.com/93427278/200621130-7cbde7df-d1d2-449c-aafa-d5c404639720.png)



![2](https://user-images.githubusercontent.com/93427278/200621153-a98cc3b5-dd84-4317-bd9f-2d37a1a2ce55.png)



![3](https://user-images.githubusercontent.com/93427278/200621216-0705583c-3a84-428d-80ea-271772851342.png)


![4](https://user-images.githubusercontent.com/93427278/200621219-4da55088-4b6d-4317-965c-b61c08727062.png)


![5](https://user-images.githubusercontent.com/93427278/200621264-dd3790d1-7b1f-4313-a5fb-a45d21e2b60a.png)



![6](https://user-images.githubusercontent.com/93427278/200621314-54c9d8ea-c343-4841-aab8-80f93a9f4033.png)


![7](https://user-images.githubusercontent.com/93427278/200621361-90a5c368-a427-4c22-87a7-07d3e5731724.png)


![8](https://user-images.githubusercontent.com/93427278/200621406-cf0825af-95db-4e67-bdbc-80c871752e9b.png)



![9](https://user-images.githubusercontent.com/93427278/200621441-3ba11f56-6c9e-43a6-9aff-91d02230ce5b.png)


![10](https://user-images.githubusercontent.com/93427278/200621467-0b672fc5-3a3a-46a7-9da6-deb729dc56c6.png)


![11](https://user-images.githubusercontent.com/93427278/200621501-e7c247ef-b790-4b58-a32a-4810d716111b.png)


![12](https://user-images.githubusercontent.com/93427278/200621542-1a828fb3-693e-45c2-81ca-b3692d303c74.png)


![13](https://user-images.githubusercontent.com/93427278/200621569-a50c2ad2-0852-4607-90c2-f2d8c429f03b.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
