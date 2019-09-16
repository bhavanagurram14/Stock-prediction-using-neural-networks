#STOCK PRICE PREDICTION- Classification Problem

#imports
from sklearn.model_selection import train_test_split     
import numpy as np

#Reading csv file using numpy
wines = np.genfromtxt("/home/komali_priya/Documents/google.csv", delimiter=",", skip_header=0)
#print("wines  ",wines)

#Normalizing
wines = wines.T
maxArray = []
for i in range (1 , len(wines)):
    maximum = max(wines[i])
    maxArray.append(maximum)
    k = 0
    for j in wines[i]:
        
        j = j/maximum
        wines[i][k] = j
        k = k+1
 
    
#print(maxArray)
wines = wines.T

X = []
for i in range(len(wines)-1):
    X.append(wines[i])
X=np.asarray(X)

#print("X is  ",X)

# Random Splitting of training and testing data 
X_train, X_test = train_test_split(X, test_size=0.4, random_state=42 )
#print("X_train is  ",X_train)

#Creating target lists
y_train = []
y_test = []
for i in X_train:
    j = int(i[0])
    j = j+1
    y_train.append(wines[j][1])
    
for i in X_test:
    j = int(i[0])
    j = j+1
    y_test.append(wines[j][1])

#print("y_train",y_train)

"""for i in range(5):
    print("X_train : " ,X_train[i] ,"Y_train :", y_train[i])"""
    
x_train = []
x_test = []

for i in X_train :
    x_train.append([i[1],i[2],i[3],i[4]])
for i in X_test :
    x_test.append([i[1],i[2],i[3],i[4]])
    
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

    
"""for i in range(5):
    print("x_train : " ,x_train[i])"""
    
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


train_target = []
for i in range(len(X_train)) :
    if (X_train[i][1] < y_train[i] ):
        train_target.append(1)     #"1" representing increase in stock price
    else:
         train_target.append(0)   #"0" representing decrease in stock price
#print("train_target  ",train_target)

test_target = []
for i in range(len(X_test)) :
    if (X_test[i][1] < y_test[i] ):
        test_target.append(1)
    else:
         test_target.append(0)
#print("test_target  ",test_target)

train_target = np.asarray(train_target)
#test_target = np.asarray(test_target)


print(len(x_train))
# Initialising the RNN

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 512, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 32))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train , epochs =100, batch_size = 32)

output = regressor.predict(x_test)
print("Output : ", output)

output01 = []
for i in output:
    if i<0.5:
        output01.append(0)
    else:
        output01.append(1)

print("Target List :\n ",test_target)
print("Output List :\n ",output01) 

#Accuracy calculation
accuracy = 0
total = len(test_target) 
print("No. of testing tuples: ",total)
for i in range(len(test_target)):
    if( test_target[i] == output01[i] ):
        accuracy = accuracy+1
percentage = (accuracy*100)/total

print("Accuracy :  ",percentage)




    


