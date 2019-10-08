
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dfx = pd.read_csv(r"C:\Users\HP\Desktop\ml\Project_1_Hardwork_paysoff\Linear_X_Train.csv")
dfy = pd.read_csv(r"C:\Users\HP\Desktop\ml\Project_1_Hardwork_paysoff\Linear_Y_Train.csv")
x = dfx.values
y = dfy.values
#print(x.shape)
#print(x)
#plt.scatter(x,y)
#plt.show()
x = x.reshape((-1,))
y = y.reshape((-1,))
#print(x.shape)
#print(x)
X = x
Y = y
#plt.scatter(X,Y)
#plt.show()

def hypothesis(x,theta):
    return(theta[0]+theta[1]*x)
    
def error(x,y,theta):
    m=x.shape[0]
    error=0
    for i in range(m):
        h=hypothesis(x[i],theta)
        error+=(h-y[i])**2
    return error

def gradient(x,y,theta):
    m=x.shape[0]
    grad=np.zeros((2,))
    for i in range(m):
        h=hypothesis(x[i],theta)
        grad[0]+=(h-y[i])
        grad[1]+=(h-y[i])*x[i]
    return grad

def gradient_descent(x,y,k=.0001):
    theta=np.zeros((2,))
    m=x.shape[0]
    itr=100
    err_list=[]
    for i in range(itr):
        grad=gradient(x,y,theta)
        e=error(x,y,theta)
        err_list.append(e)
        theta[0]=theta[0]-k*grad[0]
        theta[1]=theta[1]-k*grad[1]
    return theta,err_list

def prediction(x,theta):
    m=x.shape[0]
    Y_Predicted=[]
    for i in range(m):
        y=hypothesis(x[i],theta)
        Y_Predicted.append(y)
    Y_Predicted=np.array(Y_Predicted)
    return Y_Predicted

final_theta=np.zeros((2,))
final_theta, error_list = gradient_descent(X,Y)
#plt.plot(final_theta)
#plt.plot(error_list)
#plt.show()
xt=np.arange(-4,6)                               
plt.plot(xt,hypothesis(xt,final_theta),color='r',label="Predicted Value ") 
plt.show() 
#Y_Predicted_Values=prediction(X,final_theta)    
#print(Y_Predicted_Values)
dx = pd.read_csv(r"C:\Users\HP\Desktop\ml\Project_1_Hardwork_paysoff\Linear_X_Test.csv")
x=dx.values
#print(x.shape)
x=x.reshape((-1,))
#print(x.shape)
y=prediction(x,final_theta)    
print(y)
output=pd.DataFrame({"y":y})
output.to_csv(r"C:\Users\HP\Desktop\ml\Project_1_Hardwork_paysoff\Linear_Y_Test.csv",index=False)       
        
    
