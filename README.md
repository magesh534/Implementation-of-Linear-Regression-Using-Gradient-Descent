# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Magesh V
RegisterNumber:212222040092
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (2).txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```

## Output:
1.profit prediction
![ex 3 1](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/b516bb5c-f151-429d-bf50-75dbdd9e734a)
2.function output
![ex 3 2](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/ce1dcc68-c409-4861-93e6-f229f561f04f)
3.Gradient Descent
![ex 3 3](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/6e93a706-8ec6-44dc-8b83-d0e863191ca2)
4.Cost function using gradient descent
![ex 3 4](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/1fda2603-8672-4fcf-a079-953fb5295842)
5.Linear regression using profit prediction
![ex 3 5](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/fdcf5fa4-096f-40ce-8d41-e056db1efe41)
6.Profit prediction for a population of 35,000
![ex 3 6](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/d55826b0-a39c-43c7-a491-fa08f39bb9a5)
6.Profit prediction for a population of 35,000
![ex 3 7](https://github.com/magesh534/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135577936/dbbae8c9-90ad-4d31-9ba3-23f59f1238a3)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
