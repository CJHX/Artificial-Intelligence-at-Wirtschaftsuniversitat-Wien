import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def conv(s):
    if s == b'Iris-setosa': return 0.0
    if s == b'Iris-versicolor': return 1.0
    if s == b'Iris-virginica': return 2.0
    return -1

data = np.genfromtxt('iris.data', delimiter=',', converters={4:conv})
print(data.shape) 
print(set(data[:,4]))
print(data[:3,:])
print(data[-3:,:])

d = np.random.permutation(data[:100])
X = d[:,:4]
y = d[:,4]
print(X[:3,:])
print(y)

model = Sequential()
model.add(Dense(8, input_dim=4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=5)  