#Importing Libraries & Typesetting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd      
from matplotlib import rc
rc('text', usetex=True)
plt.rc('axes', axisbelow=True)

#dataset file import(Salary,Assets,Loan amount,Loan approval status)
dataset = pd.read_csv('C:/Users/Joy Ghosh/Desktop/ICE470 LAB/Lab Project/Project Material/BLA Dataset.csv')

#Spit the data
X = dataset.iloc[:,:-1].values  #except -1(dependent variable) index
Y = dataset.iloc[:,3].values

#Read the input and output according to this datasets
N = len(dataset)      #length of the dataset
x1 = np.array([i for i,j,k in X])  #Array of Salary
x2 = np.array([j for i,j,k in X])  #Array of Assets
x3 = np.array([k for i,j,k in X]) #Array of amount of loan
y  = np.array([l for l in Y])  #loan approval status


#No require to data preprocess

#For Plot
x3_1, x2_1, x1_1 = x3[y==1], x2[y==1], x1[y==1]   #plot when y=1 for Salary,Assets and Amount of loan
x3_0, x2_0, x1_0 = x3[y==0], x2[y==0], x1[y==0]   #plot when y=0 for Salary,Assets and Amount of loan

plt.figure(1, figsize = (8.5, 6), dpi = 250)
ax = plt.axes(projection='3d')
ax.scatter3D(x1_1, x2_1, x3_1,  c='green', depthshade=False)
ax.scatter3D(x1_0, x2_0, x3_0,  c='red', depthshade=False)

ax.set_xlabel(r"Salary (x1), $x1$")
ax.set_ylabel(r"Assets (x2), $x2$")
ax.set_zlabel(r" Loan Amount (x3), $x3$")
ax.grid()


#********************* FOR first hidden layer neuron (h0) *************************************** 
#%2. Function define(Two Function)
def p1(W1, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W1[3]*x3+ W1[2]*x2+ W1[1]*x1 +W1[0])))     #Sigmoid Function

def loss(W1):
    return -np.mean(y*np.log2(p1(W1, x1, x2, x3)) + (1-y)*np.log2(1-p1(W1, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
#GDM loop parameters
count = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess smartly
W1 = np.array([0 , 0, 0, 0])
W1[0], W1[1], W1[2], W1[3] = 1, 0.2, 0.15, 0.35
alpha = [0.1, 1e-4,1e-5, 1e-4]   #learning rate 
#GDM loop

while count <= max_iters:
    W_temp1 = W1   #store previous value
    
    dL0 = (loss([W1[0]+h, W1[1], W1[2], W1[3]]) - loss([W1[0]-h, W1[1], W1[2], W1[3]]))/(2*h)
    dL1 = (loss([W1[0], W1[1]+h, W1[2], W1[3]]) - loss([W1[0], W1[1]-h, W1[2], W1[3]]))/(2*h)
    dL2 = (loss([W1[0], W1[1], W1[2]+h, W1[3]]) - loss([W1[0], W1[1], W1[2]-h, W1[3]]))/(2*h)
    dL3 = (loss([W1[0], W1[1], W1[2], W1[3]+h]) - loss([W1[0], W1[1], W1[2], W1[3]-h]))/(2*h)
    gradLoss = np.array([dL0, dL1, dL2, dL3])   #gradient*l(Wi)
    W1 = W1 - alpha*gradLoss  # GDM rule
        
    if max(abs(W1 - W_temp1))<= tol:   
        break
    count +=1
print("Updated weight of Hidden layer(Blue lines):")
print("Weights from H0:\n\t W1[0] =", W1[0], "\n\t W1[1] =", W1[1],  "\n\t W1[2] =", W1[2],  "\n\t W1[3] =", W1[3])
#print("Weights, W1[0] = , W1[1]= ,W1[2]= ,W1[3]= ".format(W1))
#********************* FOR Second hidden layer neuron (h1) *************************************** 
#%2. Function define(Two Function)
def p2(W2, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W2[3]*x3+ W2[2]*x2+ W2[1]*x1 +W2[0])))     #Sigmoid Function

def loss(W2):
    return -np.mean(y*np.log2(p2(W2, x1, x2, x3)) + (1-y)*np.log2(1-p2(W2, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
count1 = 0

# take initial guess 
W2 = np.array([0 , 0, 0, 0])
W2[0], W2[1], W2[2], W2[3] = 1, 0.58, 0.74, 0.23
alpha1 = [0.1, 1e-4,1e-5, 1e-4]   
#GDM loop
while count1 <= max_iters:
    W_temp2 = W2   #store previous value
    
    dL10 = (loss([W2[0]+h, W2[1], W2[2], W2[3]]) - loss([W2[0]-h, W2[1], W2[2], W2[3]]))/(2*h)
    dL11 = (loss([W2[0], W2[1]+h, W2[2], W2[3]]) - loss([W2[0], W2[1]-h, W2[2], W2[3]]))/(2*h)
    dL12 = (loss([W2[0], W2[1], W2[2]+h, W2[3]]) - loss([W2[0], W2[1], W2[2]-h, W2[3]]))/(2*h)
    dL13 = (loss([W2[0], W2[1], W2[2], W2[3]+h]) - loss([W2[0], W2[1], W2[2], W2[3]-h]))/(2*h)
    gradLoss1 = np.array([dL10, dL11, dL12, dL13])   #gradient*l(Wi)
    W2 = W2 - alpha1*gradLoss1  # GDM rule
        
    if max(abs(W2 - W_temp2))<= tol:   
        break
    count1 +=1

print("Weights from H1:\n\t W2[0] =", W2[0], "\n\t W2[1] =", W2[1],  "\n\t W2[2] =", W2[2],  "\n\t W2[3] =", W2[3])

#********************* FOR Third hidden layer neuron (h2) *************************************** 
#%2. Function define(Two Function)
def p3(W3, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W3[3]*x3+ W3[2]*x2+ W3[1]*x1 +W3[0])))     #Sigmoid Function

def loss(W3):
    return -np.mean(y*np.log2(p3(W3, x1, x2, x3)) + (1-y)*np.log2(1-p3(W3, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
count2 = 0

# take initial guess 
W3 = np.array([0 , 0, 0, 0])
W3[0], W3[1], W3[2], W3[3] = 1.5, 0.40, 0.65, 0.15
alpha2 = [0.1, 1e-4,1e-5, 1e-4]   

#GDM loop
while count2 <= max_iters:
    W_temp3 = W3   #store previous value
    
    dL20 = (loss([W3[0]+h, W3[1], W3[2], W3[3]]) - loss([W3[0]-h, W3[1], W3[2], W3[3]]))/(2*h)
    dL21 = (loss([W3[0], W3[1]+h, W3[2], W3[3]]) - loss([W3[0], W3[1]-h, W3[2], W3[3]]))/(2*h)
    dL22 = (loss([W3[0], W3[1], W3[2]+h, W3[3]]) - loss([W3[0], W3[1], W3[2]-h, W3[3]]))/(2*h)
    dL23 = (loss([W3[0], W3[1], W3[2], W3[3]+h]) - loss([W3[0], W3[1], W3[2], W3[3]-h]))/(2*h)
    gradLoss2 = np.array([dL20, dL21, dL22, dL23])   #gradient*l(Wi)
    W3 = W3 - alpha2*gradLoss2  # GDM rule
        
    if max(abs(W3 - W_temp3))<= tol:   
        break
    count2 +=1

print("Weights from H2:\n\t W3[0] =", W3[0], "\n\t W3[1] =", W3[1],  "\n\t W3[2] =", W3[2],  "\n\t W3[3] =", W3[3])

#********************* FOR Fourth hidden layer neuron (h2) *************************************** 
#%2. Function define(Two Function)
def p4(W4, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W4[3]*x3+ W4[2]*x2+ W4[1]*x1 +W4[0])))     #Sigmoid Function

def loss(W4):
    return -np.mean(y*np.log2(p4(W4, x1, x2, x3)) + (1-y)*np.log2(1-p4(W4, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
count3 = 0

# take initial guess 
W4 = np.array([0 , 0, 0, 0])
W4[0], W4[1], W4[2], W4[3] = 2, 0.35, 0.55, 0.20
alpha3 = [0.1, 1e-4,1e-5, 1e-4]   #learning rate random(small value) For [W3, W2, W1, W0]

#GDM loop
while count3 <= max_iters:
    W_temp4 = W4   #store previous value
    
    dL30 = (loss([W4[0]+h, W4[1], W4[2], W4[3]]) - loss([W4[0]-h, W4[1], W4[2], W4[3]]))/(2*h)
    dL31 = (loss([W4[0], W4[1]+h, W4[2], W4[3]]) - loss([W4[0], W4[1]-h, W4[2], W4[3]]))/(2*h)
    dL32 = (loss([W4[0], W4[1], W4[2]+h, W4[3]]) - loss([W4[0], W4[1], W4[2]-h, W4[3]]))/(2*h)
    dL33 = (loss([W4[0], W4[1], W4[2], W4[3]+h]) - loss([W4[0], W4[1], W4[2], W4[3]-h]))/(2*h)
    gradLoss3 = np.array([dL30, dL31, dL32, dL33])   #gradient*l(Wi)
    W4 = W4 - alpha3*gradLoss3  # GDM rule
        
    if max(abs(W4 - W_temp4))<= tol:   
        break
    count3 +=1

print("Weights from H3:\n\t W4[0] =", W4[0], "\n\t W4[1] =", W4[1],  "\n\t W4[2] =", W4[2],  "\n\t W4[3] =", W4[3])

#********************* FOR Classifier Neuron layer *************************************** 
#set a variable for hidden layer p-value output
p1 = p1(W1, x1, x2, x3)
p2 = p2(W2, x1, x2, x3)
p3 = p3(W3, x1, x2, x3)
p4 = p4(W4, x1, x2, x3)

#%2. Function define(Two Function)
def P1(W_c, p1, p2, p3, p4):
    return 0.999999/(1+np.exp(-(W_c[4]*p4+ W_c[3]*p3+ W_c[2]*p2+ W_c[1]*p1 +W_c[0])))     #Sigmoid Function

def loss(W_c):
    return -np.mean(y*np.log2(P1(W_c, p1, p2, p3, p4)) + (1-y)*np.log2(1-P1(W_c, p1, p2, p3, p4)))   #Mean Loss Function

# GDM use for Training
#GDM loop parameters
count4 = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess smartly
W_c = np.array([0 , 0, 0, 0, 0])
W_c[0], W_c[1], W_c[2], W_c[3], W_c[4] = 2.5, 0.85, 0.25, 0.55, 0.10
alpha = [4, 1e-5,1e-5, 1e-4, 1e-3]   

#GDM loop

while count4 <= max_iters:
    W_temp_c = W_c   #store previous value
    
    dL0_1 = (loss([W_c[0]+h, W_c[1], W_c[2], W_c[3], W_c[4]]) - loss([W_c[0]-h, W_c[1], W_c[2], W_c[3], W_c[4]]))/(2*h)
    dL1_2 = (loss([W_c[0], W_c[1]+h, W_c[2], W_c[3], W_c[4]]) - loss([W_c[0], W_c[1]-h, W_c[2], W_c[3], W_c[4]]))/(2*h)
    dL2_3 = (loss([W_c[0], W_c[1], W_c[2]+h, W_c[3], W_c[4]]) - loss([W_c[0], W_c[1], W_c[2]-h, W_c[3], W_c[4]]))/(2*h)
    dL3_4 = (loss([W_c[0], W_c[1], W_c[2], W_c[3]+h, W_c[4]]) - loss([W_c[0], W_c[1], W_c[2], W_c[3]-h, W_c[4]]))/(2*h)
    dL4_5 = (loss([W_c[0], W_c[1], W_c[2], W_c[3], W_c[4]+h]) - loss([W_c[0], W_c[1], W_c[2], W_c[3], W_c[4]-h]))/(2*h)
    gradLoss = np.array([dL0_1, dL1_2, dL2_3, dL3_4, dL4_5])   #gradient*l(Wi)
    W_c = W_c - alpha*gradLoss  # GDM rule
    
    if max(abs(W_c - W_temp_c))<= tol:   
        break
    count4 +=1

print("\nUpdated weight of Classifier Neuron(Green lines):-------->>")
print("Weights from Classifier Neuron:\n\t W_c[0] =", W_c[0], "\n\t W_c[1] =", W_c[1],  "\n\t W_c[2] =", W_c[2],  "\n\t W_c[3] =", W_c[3],  "\n\t W_c[4] =", W_c[4])


#-------------------------Repeat first hidden layer--------------------------

#%2. Function define(Two Function)
def p5(W5, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W5[3]*x3+ W5[2]*x2+ W5[1]*x1 +W5[0])))     #Sigmoid Function

def loss(W5):
    return -np.mean(y*np.log2(p5(W5, x1, x2, x3)) + (1-y)*np.log2(1-p5(W5, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
#GDM loop parameters
count5 = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess 
W5 = np.array([0 , 0, 0, 0])
W5[0], W5[1], W5[2], W5[3] = W1  #W1 previous updated weight
alpha = [1.5, 1e-4,1e-5, 1e-4]   
#GDM loop

while count5 <= max_iters:
    W_temp5 = W5   #store previous value
    
    dL_0 = (loss([W5[0]+h, W5[1], W5[2], W5[3]]) - loss([W5[0]-h, W5[1], W5[2], W5[3]]))/(2*h)
    dL_1 = (loss([W5[0], W5[1]+h, W5[2], W5[3]]) - loss([W5[0], W5[1]-h, W5[2], W5[3]]))/(2*h)
    dL_2 = (loss([W5[0], W5[1], W5[2]+h, W5[3]]) - loss([W5[0], W5[1], W5[2]-h, W5[3]]))/(2*h)
    dL_3 = (loss([W5[0], W5[1], W5[2], W5[3]+h]) - loss([W5[0], W5[1], W5[2], W5[3]-h]))/(2*h)
    gradLoss5 = np.array([dL_0, dL_1, dL_2, dL_3])   #gradient*l(Wi)
    W5 = W5 - alpha*gradLoss5  # GDM rule
        
    if max(abs(W5 - W_temp5))<= tol:   
        break
    count5 +=1
    
print("\nUpdated weight of Classifier Neuron(Green lines):------->")
print("Updated Weights from H0:\n\t W5[0] =", W5[0], "\n\t W5[1] =", W5[1],  "\n\t W5[2] =", W5[2],  "\n\t W5[3] =", W5[3])

#------------------------------Reapeat for second hidden layer-------------------
#%2. Function define(Two Function)
def p6(W6, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W6[3]*x3+ W6[2]*x2+ W6[1]*x1 +W6[0])))     #Sigmoid Function

def loss(W6):
    return -np.mean(y*np.log2(p6(W6, x1, x2, x3)) + (1-y)*np.log2(1-p6(W6, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
#GDM loop parameters
count6 = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess smartly
W6 = np.array([0 , 0, 0, 0])
W6[0], W6[1], W6[2], W6[3] = W2
alpha = [2, 1e-4,1e-5, 1e-4]   
#GDM loop

while count6 <= max_iters:
    W_temp6 = W6   #store previous value
    
    dL_10 = (loss([W6[0]+h, W6[1], W6[2], W6[3]]) - loss([W6[0]-h, W6[1], W6[2], W6[3]]))/(2*h)
    dL_11 = (loss([W6[0], W6[1]+h, W6[2], W6[3]]) - loss([W6[0], W6[1]-h, W6[2], W6[3]]))/(2*h)
    dL_12 = (loss([W6[0], W6[1], W6[2]+h, W6[3]]) - loss([W6[0], W6[1], W6[2]-h, W6[3]]))/(2*h)
    dL_13 = (loss([W6[0], W6[1], W6[2], W6[3]+h]) - loss([W6[0], W6[1], W6[2], W6[3]-h]))/(2*h)
    gradLoss6 = np.array([dL_10, dL_11, dL_12, dL_13])   #gradient*l(Wi)
    W6 = W6 - alpha*gradLoss6  # GDM rule
        
    if max(abs(W6 - W_temp6))<= tol:   
        break
    count6 +=1

print("Updated Weights from H1:\n\t W6[0] =", W6[0], "\n\t W6[1] =", W6[1],  "\n\t W6[2] =", W6[2],  "\n\t W6[3] =", W6[3])

#----------------------Reapeat for Third hidden layer-------------------
#%2. Function define(Two Function)
def p7(W7, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W7[3]*x3+ W7[2]*x2+ W7[1]*x1 +W7[0])))     #Sigmoid Function

def loss(W7):
    return -np.mean(y*np.log2(p7(W7, x1, x2, x3)) + (1-y)*np.log2(1-p7(W7, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
#GDM loop parameters
count7 = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess smartly
W7 = np.array([0 , 0, 0, 0])
W7[0], W7[1], W7[2], W7[3] = W3
alpha = [2.5, 1e-4,1e-5, 1e-4]   
#GDM loop

while count7 <= max_iters:
    W_temp7 = W7   #store previous value
    
    dL_20 = (loss([W7[0]+h, W7[1], W7[2], W7[3]]) - loss([W7[0]-h, W7[1], W7[2], W7[3]]))/(2*h)
    dL_21 = (loss([W7[0], W7[1]+h, W7[2], W7[3]]) - loss([W7[0], W7[1]-h, W7[2], W7[3]]))/(2*h)
    dL_22 = (loss([W7[0], W7[1], W7[2]+h, W7[3]]) - loss([W7[0], W7[1], W7[2]-h, W7[3]]))/(2*h)
    dL_23 = (loss([W7[0], W7[1], W7[2], W7[3]+h]) - loss([W7[0], W7[1], W7[2], W7[3]-h]))/(2*h)
    gradLoss7 = np.array([dL_20, dL_21, dL_22, dL_23])   #gradient*l(Wi)
    W7 = W7 - alpha*gradLoss7  # GDM rule
        
    if max(abs(W7 - W_temp7))<= tol:   
        break
    count7 +=1

print("Updated Weights from H2:\n\t W7[0] =", W7[0], "\n\t W7[1] =", W7[1],  "\n\t W7[2] =", W7[2],  "\n\t W7[3] =", W7[3])

#----------------------Reapeat Final neuron of hidden layer-------------------
#%Function define(Two Function)
def P(W, x1, x2, x3):
    return 0.999999/(1+np.exp(-(W[3]*x3+ W[2]*x2+ W[1]*x1 +W[0])))     #Sigmoid Function

def loss(W):
    return -np.mean(y*np.log2(P(W, x1, x2, x3)) + (1-y)*np.log2(1-P(W, x1, x2, x3)))   #Mean Loss Function

#3: GDM use for Training
#GDM loop parameters
count8 = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess smartly
W = np.array([0 , 0, 0, 0])
W[0], W[1], W[2], W[3] = W4
alpha = [4, 1e-4,1e-5, 1e-4]   
#GDM loop

while count8 <= max_iters:
    W_temp = W   #store previous value
    
    dL_30 = (loss([W[0]+h, W[1], W[2], W[3]]) - loss([W[0]-h, W[1], W[2], W[3]]))/(2*h)
    dL_31 = (loss([W[0], W[1]+h, W[2], W[3]]) - loss([W[0], W[1]-h, W[2], W[3]]))/(2*h)
    dL_32 = (loss([W[0], W[1], W[2]+h, W[3]]) - loss([W[0], W[1], W[2]-h, W[3]]))/(2*h)
    dL_33 = (loss([W[0], W[1], W[2], W[3]+h]) - loss([W[0], W[1], W[2], W[3]-h]))/(2*h)
    gradLoss8 = np.array([dL_30, dL_31, dL_32, dL_33])   #gradient*l(Wi)
    W = W - alpha*gradLoss8  # GDM rule
        
    if max(abs(W - W_temp))<= tol:   
        break
    count8 +=1

print("Updated Weights from H3:\n\t W[0] =", W[0], "\n\t W[1] =", W[1],  "\n\t W[2] =", W[2],  "\n\t W[3] =", W[3])


#********************* Repeat FOR Classifier Neuron layer ********************* 
#set a variable for hidden layer p-value output
P_1 = p5(W5, x1, x2, x3)
P_2 = p6(W6, x1, x2, x3)
P_3 = p7(W7, x1, x2, x3)
P_4 = P(W, x1, x2, x3)

#%2. Function define(Two Function)
def P_(W9, P_1, P_2, P_3, P_4):
    return 0.999999/(1+np.exp(-(W9[4]*P_4+ W9[3]*P_3+ W9[2]*P_2+ W9[1]*P_1 +W9[0])))     #Sigmoid Function

def loss(W9):
    return -np.mean(y*np.log2(P_(W9, P_1, P_2, P_3, P_4)) + (1-y)*np.log2(1-P_(W9, P_1, P_2, P_3, P_4)))   #Mean Loss Function

# GDM use for Training
#GDM loop parameters
count9 = 0
max_iters = 4000
tol = 1e-3
#define h
h = 1e-4  #h = small number

# take initial guess smartly
W9 = np.array([0 , 0, 0, 0, 0])
W9[0], W9[1], W9[2], W9[3], W9[4] = W_c
alpha = [4, 0.07,0.007, 0.0007, 1e-1]   

#GDM loop

while count9 <= max_iters:
    W_temp9 = W9   #store previous value
    
    dL0_11 = (loss([W9[0]+h, W9[1], W9[2], W9[3], W9[4]]) - loss([W9[0]-h, W9[1], W9[2], W9[3], W9[4]]))/(2*h)
    dL1_21 = (loss([W9[0], W9[1]+h, W9[2], W9[3], W9[4]]) - loss([W9[0], W9[1]-h, W9[2], W9[3], W9[4]]))/(2*h)
    dL2_31 = (loss([W9[0], W9[1], W9[2]+h, W9[3], W9[4]]) - loss([W9[0], W9[1], W9[2]-h, W9[3], W9[4]]))/(2*h)
    dL3_41 = (loss([W9[0], W9[1], W9[2], W9[3]+h, W9[4]]) - loss([W9[0], W9[1], W9[2], W9[3]-h, W9[4]]))/(2*h)
    dL4_51 = (loss([W9[0], W9[1], W9[2], W9[3], W9[4]+h]) - loss([W9[0], W9[1], W9[2], W9[3], W9[4]-h]))/(2*h)
    gradLoss9 = np.array([dL0_11, dL1_21, dL2_31, dL3_41, dL4_51])   #gradient*l(Wi)
    W9 = W9 - alpha*gradLoss9  # GDM rule
    
    if max(abs(W9 - W_temp9))<= tol:   
        break
    count9 +=1

print("\nUpdated weight of Classifier Neuron(Green lines):------>")
print("iteration Count = ", count9)
print("Updated Weights from Classifier Neuron:\n\t W[0] =", W9[0], "\n\t W[1] =", W9[1],  "\n\t W[2] =", W9[2],  "\n\t W[3] =", W9[3],  "\n\t W[4] =", W9[4])
print("\nFinal Cross Entropy Loss = ", "{:.8f}".format(loss(W9)), 'bits')

# Accuracy Test compare to prediction value
#Condition
p_pred9 = P_(W9, P_1, P_2, P_3, P_4)
y_pred9 = np.array([1 if pi >= 0.5 else 0 for pi in p_pred9])
n_error9 = sum(abs(y-y_pred9))   # number of 1 probability
acc = (N - n_error9)/N  #Accuracy ((N-n_error) number correct)

print("Final Accuracy = ", acc*100, "%")

#Curve Fitting(Plot)
x1r = np.linspace(min(x1) ,max(x2), 200)
x2r = np.linspace(min(x2) ,max(x2), 200)  
x3r = (-W[2]/W[3])*x2r +(-W[1]/W[3])*x1r + (-W[0]/W[3])  #boundary condition 

x3_1, x2_1, x1_1 = x3[y==1], x2[y==1], x1[y==1]   #plot when y=1 for Salary,Assets and Amount of loan
x3_0, x2_0, x1_0 = x3[y==0], x2[y==0], x1[y==0]   #plot when y=0 for Salary,Assets and Amount of loan

plt.figure(2, figsize = (8.5, 6), dpi = 250)
ax = plt.axes(projection='3d')
ax.scatter3D(x1_1, x2_1, x3_1,  c='green', depthshade=False)
ax.scatter3D(x1_0, x2_0, x3_0,  c='red', depthshade=False)

ax.set_xlabel(r"Salary (x1), $x1$")
ax.set_ylabel(r"Assets (x2), $x2$")
ax.set_zlabel(r" Loan Amount (x3), $x3$")
ax.grid()

#Scatter Fitting Final result plot
ax.plot3D(x1r, x2r, x3r, 'b--')

#Test given the number of 6 customers loan approval Probability
customer = 0 
for i in range(6):
    customer +=1
    print("\nCustomer number: ",customer)          
    Salary = float(input("Enter the Salary = "))
    Assets = float(input("Enter the Assets = "))
    Loan_Amount = float(input("Enter the Loan amount = "))
    
    
    print("Probability of Loan Approval = ", P(W,Salary,Assets,Loan_Amount)*100, "%")
    
    