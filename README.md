# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHAIK MUFEEZUR RAHAMAN
RegisterNumber: 212221043007
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
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
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) 
*/

## Output:

![271904835-16ff60b8-2e4f-4a97-a8ea-a4e3dd3f59d1](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/c26db8ac-dcde-421b-8bcf-0cff08c19b51)

![271904842-2bad8ba3-d85d-47de-88f0-fc2ae7181f45](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/a2a5082a-2372-41ab-a4e6-6e16f79abc3f)


![271904851-868ffe64-5594-47a7-ba87-b1143cff5614](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/c8f27250-74b3-4032-bc91-b1bdb906cc72)

![271904863-e7a5cb06-248d-4835-a4ea-4eb9e2da277e](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/103991db-6b03-42b5-b3e9-489c250528aa)

![271904874-c1c23a87-accc-422c-8815-976a74a3b576](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/c7fabbe1-a9fd-4747-8f78-c2bfd9878eab)

![271904882-7974ab8f-8dc4-43d4-b712-66a2e9ccb9bc](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/32c9a106-b9c2-4aee-a9ef-a5811bf21dd1)

![271904886-c9bad502-25ac-4b02-8239-04e8fd45e821](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/fe7e4b4a-e660-4f11-a48d-8122e3d02689)

![271904895-34ca7352-6af4-4cbd-b519-59d3cdeaa84b](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/6f6118d1-06a9-4169-bcbf-c9fa1ed78072)

![271904915-f24974e1-21e7-4d0f-88c2-943f5ce682fa](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/91b99d97-d5ce-4c57-987d-b8fbb424c445)


![271905047-c7fa7ac5-3c31-4e62-a478-7c36ccb726eb](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/b9816585-c729-4c93-b903-f3d5d6a90329)

![271905068-ef37e93d-818d-4512-80ce-be4e13f0cbce](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/3ce67bf6-fb2d-44e5-ba0c-e68201312b3f)

![271905079-347f7837-9501-4598-9471-3df1a7dce2ce](https://github.com/githubmufeez45/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/134826568/261408f9-aebb-441f-8cc0-ec7358dfa1ec)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
