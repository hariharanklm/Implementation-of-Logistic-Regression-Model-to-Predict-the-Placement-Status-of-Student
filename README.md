# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: K.Jhansi
RegisterNumber:  212221230045
*/
```
```
import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```
## Output:
### Original Data:
<img width="655" alt="output1" src="https://user-images.githubusercontent.com/117884666/202349242-0eae951d-3d88-4646-af77-7d7401b8a246.png">

### After removing:
<img width="655" alt="output2" src="https://user-images.githubusercontent.com/117884666/202349281-5cdf3222-d236-48cf-9464-2d00de90333c.png">

### Null Data:
<img width="347" alt="output3" src="https://user-images.githubusercontent.com/117884666/202349327-d3140b03-9ab3-4bbb-8825-280809042668.png">

### Label Encoder:
<img width="648" alt="output4" src="https://user-images.githubusercontent.com/117884666/202349360-d1fad749-cad4-45f6-b452-b639488d00f1.png">

### X:
<img width="648" alt="output5" src="https://user-images.githubusercontent.com/117884666/202349397-13ddd034-640d-4a0d-9b80-d40e829498eb.png">

### Y:
<img width="648" alt="output6" src="https://user-images.githubusercontent.com/117884666/202349451-6b400bf1-2442-42f9-b626-2515a6890628.png">

### Y_prediction:
<img width="648" alt="output7" src="https://user-images.githubusercontent.com/117884666/202349482-eeebf89b-8723-49ab-b2ee-f2799032d998.png">

### Accuracy:
<img width="648" alt="outpit8" src="https://user-images.githubusercontent.com/117884666/202349521-b5dda1e4-24f4-450b-8d25-cab697917455.png">

### Cofusion:
<img width="460" alt="output9" src="https://user-images.githubusercontent.com/117884666/202349566-cfb1debe-5ad0-4d63-9058-37fb909276c6.png">

### Classification:
<img width="648" alt="output10" src="https://user-images.githubusercontent.com/117884666/202349583-7473e1af-27fc-4171-9e07-e84d48626af6.png">



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
