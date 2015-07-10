from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle
import math
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    A = np.column_stack((X,y))
    grp = np.unique(A[:,2])
    B = np.array([0.0,0.0,0.0,0.0,0.0])
    C = np.array([0.0,0.0,0.0,0.0,0.0])
    A = A[A[:,2].argsort()]
    uniq, inv = np.unique(y, return_inverse=True)
    n = np.bincount(inv)
    for idx, val in enumerate(A[:,2]):
        for idx1, val1 in enumerate(grp):
            if (val == val1):
                B[idx1]+=A[idx,0]
                C[idx1]+=A[idx,1]
    means = np.hstack((B/n, C/n))
    
    gmean = X.mean(axis=0)
    A[:,0] = A[:,0] - gmean[0];
    A[:,1] = A[:,1] - gmean[1];
    
    covmat = np.cov(A[:,:2], rowvar = 0)
    
    return means,covmat
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    A = np.column_stack((X,y))
    grp = np.unique(A[:,2])
    B = np.array([0.0,0.0,0.0,0.0,0.0])
    C = np.array([0.0,0.0,0.0,0.0,0.0])
    A = A[A[:,2].argsort()]
    uniq, inv = np.unique(y, return_inverse=True)
    n = np.bincount(inv)
    for idx, val in enumerate(A[:,2]):
        for idx1, val1 in enumerate(grp):
            if (val == val1):
                B[idx1]+=A[idx,0]
                C[idx1]+=A[idx,1]
    means = np.hstack((B/n, C/n))
    
    gmean = X.mean(axis=0)
    A[:,0] = A[:,0] - gmean[0];
    A[:,1] = A[:,1] - gmean[1];
    #print A
    #cov = np.cov(A[:,:2], rowvar = 0)
    covmats = []
    p=0
    for val in n:
        ar = np.matrix(A[p:val+p,:2])
        ar1 = np.cov(ar, rowvar=0)
        covmats.append(ar1)
        p=val+p
    return means,covmats
def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    k = np.amax(y);
    k = int(k);
    maxval = 0;
    pred = 0;
    m = means;
    m1 = np.ndarray([]);
    m1 = np.reshape(m,(2,5)).T;
    
    prediction = np.zeros((100,1));
    
    temp = 0;      
                 
    for i in range(0, ytest.shape[0]):
        for j in range(1, k+1):
            tmp = np.mat(m1[j-1,:]);
        
            temp1 = (1/ ((2*3.14168)**(Xtest.shape[1]/2.0)) * math.sqrt((np.linalg.det(covmat))))
            temp2 = math.exp((-0.5)*(Xtest[i]-tmp) * covmat * (Xtest[i]-tmp).T);
            temp = temp1*temp2
        
            if temp > maxval:
                maxval = temp;
                pred = j; 
        maxval = 0;               
        prediction[i] = pred;
        
#----------------------------begin setup for plotting discriminant boundary linear------------------------------------------------------------------------------
        
    x1_lin = np.sort(Xtest[:,0])
    x2_lin = np.sort(Xtest[:,1])
    X1_lin, X2_lin = np.meshgrid(x1_lin,x2_lin)
    n_lin = x1_lin.shape[0]*x2_lin.shape[0]
    D_lin = np.zeros((n_lin,2))
    D_lin[:,0] = X1_lin.ravel();
    D_lin[:,1] = X2_lin.ravel();
    
    
    
    prediction_dis_lin = np.zeros((10000,1));
    
    temp=0;
    maxval=0;
    pred=0;
    
    for i in range(0, D_lin.shape[0]):
        for j in range(1, k+1):
            tmp = np.mat(m1[j-1,:]);
        
            temp1 = (1/ ((2*3.14168)**(D_lin.shape[1]/2.0)) * math.sqrt((np.linalg.det(covmat))))
            temp2 = math.exp((-0.5)*(D_lin[i]-tmp) * covmat * (D_lin[i]-tmp).T);
            temp = temp1*temp2
        
            if temp > maxval:
                maxval = temp;
                pred = j; 
        maxval = 0;               
        prediction_dis_lin[i] = pred;
    
    labels_dis_lin = prediction_dis_lin.reshape(x1_lin.shape[0], x2_lin.shape[0]);
    plt.figure(1)
    plt.title('LDA Discriminant Boundary')
    plt.contourf(x1_lin,x2_lin,labels_dis_lin)
    plt.show()
#--------------------------------------------------finish plotting discriminant boundary-------------------------------------------------------
#--------------------------------------------------calculate accuracy percentage---------------------------------------------------------------------        
        
    match = 0.0;
    for i in range(0, prediction.shape[0]):
        #print prediction[i] , ytest[i]
        if ytest[i] == prediction[i]:
            match = match + 1;
    acc = match / ytest.shape[0];
    acc = acc*100
    
    return acc
def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    k = np.amax(y);
    k = int(k);
    maxval = 0;
    pred = 0;
    m = means;
    m1 = np.ndarray([]);
    m1 = np.reshape(m,(2,5)).T;
    
    prediction = np.zeros((100,1));
    
    temp = 0;      
                 
    for i in range(0, ytest.shape[0]):
        for j in range(1, k+1):
            tmp = np.mat(m1[j-1,:]);
        
            temp1 = (1/ (2*3.14168)**(Xtest.shape[1]/2.0) * math.sqrt((np.linalg.det(covmats[j-1]))))
            temp2 = math.exp((-0.5)*(Xtest[i]-tmp) * covmats[j-1]* (Xtest[i]-tmp).T);
            temp = temp1*temp2
        
            if temp > maxval:
                maxval = temp;
                pred = j; 
        maxval = 0;               
        prediction[i] = pred;
        
#--------------------------------------------------------------begin setup for plotting discriminant boundary quadratic------------------------------
         
    x1 = np.sort(Xtest[:,0])
    x2 = np.sort(Xtest[:,1])
    X1, X2 = np.meshgrid(x1,x2)
    n = x1.shape[0]*x2.shape[0]
    D = np.zeros((n,2))
    D[:,0] = X1.ravel();
    D[:,1] = X2.ravel();
    
    
    
    prediction_dis = np.zeros((10000,1));
    
    temp=0;
    maxval=0;
    pred=0;
    
    for i in range(0, D.shape[0]):
        for j in range(1, k+1):
            tmp = np.mat(m1[j-1,:]);
        
            temp1 = (1/ (2*3.14168)**(D.shape[1]/2.0) * math.sqrt((np.linalg.det(covmats[j-1]))))
            temp2 = np.exp((-0.5)*(D[i]-tmp) * covmats[j-1]* (D[i]-tmp).T);
            temp = temp1*temp2
        
            if temp > maxval:
                maxval = temp;
                pred = j; 
        maxval = 0;               
        prediction_dis[i] = pred;
    
    labels_dis = prediction_dis.reshape(x1.shape[0], x2.shape[0]);
    plt.figure(2)
    plt.title('QDA Discriminant Boundary')
    plt.contourf(x1,x2,labels_dis)
    plt.show()
    
#---------------------------------------------------finish plotting discriminant boundary quadratic---------------------------------------------------------------
#---------------------------------------------------calculate percentage accuracy for qda------------------------------------------------------------------------
    
    match = 0.0;
    for i in range(0, prediction.shape[0]):
        #print prediction[i] , ytest[i]
        if ytest[i] == prediction[i]:
            match = match + 1;
    acc = match / ytest.shape[0];
    acc = acc*100
    return acc
def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   
    tmp1 = np.dot(X.T,X)
    tmp2 = np.dot(np.linalg.inv(tmp1),X.T)
    w = np.dot(tmp2, y)
    
    return w
def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1
    # IMPLEMENT THIS METHOD
    n = X.shape[0]
    d = X.shape[1]                                                                
    Id = np.identity(d) 
    temp1 = np.linalg.inv(lambd*n*Id+np.dot(X.T,X))
    temp2 = np.dot(X,temp1)
    w = np.dot(temp2.T,y)
                                             
    return w
def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    rmse = 0
    n = Xtest.shape[0]
    for i in range(0,n):
        rmse += (ytest[i] -  np.dot(w.T,Xtest[i]))**2
    rmse = np.sqrt(rmse)/n
    return rmse
def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  
    # IMPLEMENT THIS METHOD
    n =len(X)
    w = np.mat(w).T
    #w = np.zeros((X_i.shape[1],1))
    
    tmp1 = (y - np.dot(X,w))
    tmp2 = (lambd*np.dot(w.T,w))/2
    error = (np.dot(tmp1.T,tmp1))/(2*n) + tmp2
    
    delJ = (-(np.dot(y.T,X)) + (np.dot(w.T,np.dot(X.T,X))))/n + lambd*w.T
    error_grad1 = np.array(delJ)
    error_grad = np.ndarray.flatten(error_grad1)
                                                                                                                                                                        
    return error, error_grad
def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
    Xd = np.ones((x.shape[0],1)) 
    for i in range(1,p+1):
        tmp = np.power(x,i)
        Xd = np.column_stack((Xd,tmp))
    
    return Xd
# Main script


# Problem 1


# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            
# LDA

means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA

means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('C:\Users\Daniel\Desktop\Machine Learning Project 2\diabetes.pickle','rb'))
   
# add intercept

X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3


k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    rmses3_train[i] = testOLERegression(w_l,X_i,y)
    i = i + 1
    


#-------------------------------plot weight magnitudes from Problem 2 and 3 as scatter chart----------------------------------------------

nums =np.linspace(1,65,65)

plt.figure(3)
plt.scatter(nums,w_i.T,color='blue',s=5,edgecolor='none',label='Problem 2 weights')
plt.scatter(nums,w_l.T,color='red',s=5,edgecolor='none',label='Problem 3 weights')
plt.ylim(w_i[np.argmin(w_i)],w_i[np.argmax(w_i)])
plt.grid(True)
plt.ylabel('Weight Magnitudes for Problem 2 and 3')
plt.legend(loc ='upper right')
plt.show()
#--------------------------------------finish plotting weight magnitudes------------------------------------------------------------------------

#--------------------------------------plot graph for problem 3 with lambda optimum-----------------------------------------------------
plt.figure(4)
plt.plot(lambdas,rmses3)
plt.title('Problem 3 Test Data')
plt.xlabel('LAMBDA')
plt.ylabel('Root Mean Squared Error')
plt.text(0.00024,3.75,'LAMBDA OPTIMUM=(0.00024,3.77582332)')
plt.show()
lambda_opt = lambdas[np.argmin(rmses3)]

plt.figure(5)
plt.plot(lambdas,rmses3_train)
plt.title('Problem 3 Train Data')
plt.xlabel('LAMBDA')
plt.ylabel('Root Mean Squared Error(train Data)')
plt.show()
lambda_opt = lambdas[np.argmin(rmses3_train)]

#--------------------------------------------finish graph plot of lambda optimum---------------------------------------------------------

# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
rmses4_train =np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l.x = w_l.x.T
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    rmses4_train[i] = testOLERegression(w_l_1,X_i,y)
    i = i + 1
    
#---------------------------------------------------start plotting graph for Problem 4----------------------------------------------------------------------
plt.figure(6)
plt.title('Problem 4 Plot Test Data')
plt.plot(lambdas,rmses4)
plt.xlabel('Regularization Parameter Lambda')
plt.ylabel('RMSE for Gradient Descent')
plt.show()

plt.figure(7)
plt.title('Problem 4 Plot Train Data')
plt.plot(lambdas,rmses4_train)
plt.xlabel('Regularization Parameter Lambda')
plt.ylabel('RMSE for Gradient Descent')
plt.show()

#------------------------------------------------------finish graph for problem 4-------------------------------------------------------------------------------

# Problem 5

pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    #print Xd.T.shape, y.shape
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

#---------------------------------------------------start plotting graph for Problem 5--------------------------------------------------------------------
plt.figure(8)
plt.show(plt.plot(range(pmax),rmses5))
plt.xlabel('Optimal P')
plt.ylabel('RMSE for Problem 5')
plt.legend(('No Regularization','Regularization'))

#------------------------------------------------------finish graph for problem 5-------------------------------------------------------------------------------