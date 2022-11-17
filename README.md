# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.  LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by : Hariharan M
RegisterNumber : 212221220015
*/

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1["gender"])
data1['ssc_b']=le.fit_transform(data1["ssc_b"])
data1['hsc_b']=le.fit_transform(data1["hsc_b"])
data1['hsc_s']=le.fit_transform(data1["hsc_s"])
data1['degree_t']=le.fit_transform(data1["degree_t"])
data1['workex']=le.fit_transform(data1["workex"])
data1['specialisation']=le.fit_transform(data1["specialisation"])
data1['status']=le.fit_transform(data1["status"])
print(data1)

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### Original data(first five columns):
![output1](https://user-images.githubusercontent.com/117884666/202350920-91230236-3501-467d-a41a-3cc56f8bc7a1.png)

### Data after dropping unwanted columns(first five):
![output2](https://user-images.githubusercontent.com/117884666/202350947-5c93f20c-5cdb-4eac-a543-351d97258c19.png)

### Checking the presence of null values:
![output3](https://user-images.githubusercontent.com/117884666/202351002-33e0d443-1277-4b10-ae6a-e7372ed4a0e9.png)

### Checking the presence of duplicated values:
![output4](https://user-images.githubusercontent.com/117884666/202351028-6b348fdb-e144-40e8-881f-07ad35a35f08.jpg)

### Data after Encoding:
![output5](https://user-images.githubusercontent.com/117884666/202351090-6e2a568c-c90d-45cf-b55c-cd6691101ac3.jpg)

### X Data:
![output6](https://user-images.githubusercontent.com/117884666/202351217-16bcf798-c9a1-4fcf-a9c2-960c6dbad871.jpg)

### Y Data:
![output7](https://user-images.githubusercontent.com/117884666/202351248-b6334fe4-ba38-4909-ae14-3ad70470d546.jpg)

### Predicted Values:
![output8](https://user-images.githubusercontent.com/117884666/202351308-9de792ca-0695-46c6-a56a-86773a733cca.jpg)

### Accuracy Score:
![output9](https://user-images.githubusercontent.com/117884666/202351381-aa411c93-73db-40b8-9b44-e34448807350.jpg)

### Confusion Matrix:
![output10](https://user-images.githubusercontent.com/117884666/202351428-575fd0a8-26e5-4685-80e3-d83a6edaf05c.jpg)

### Classification Report:
![output11](https://user-images.githubusercontent.com/117884666/202351456-986e4c85-a62f-4ada-8304-29767391f8e8.jpg)

### Predicting output from Regression Model:
![output12](https://user-images.githubusercontent.com/117884666/202351558-c079fd82-35f6-4206-b9f3-21b5afa6d07a.jpg)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
