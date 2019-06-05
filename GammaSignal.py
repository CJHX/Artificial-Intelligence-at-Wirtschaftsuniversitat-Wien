import numpy as np
import io
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, class_weight
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils import np_utils

def conv(s):
    if s == b'g': return 0.0
    if s == b'h': return 1.0
    return -1

data = np.genfromtxt('magic04.data', delimiter=',', converters={10:conv})
data = data[~np.isnan(data).any(axis=1)]
# ensuring that the data is correctly imported
# print(data.shape) 
# print(set(data[:,10]))
# print(data[:3,:])
# print(data[-3:,:])

# I will be using all the data from the dataset
d = np.random.permutation(data)
X = d[:,:10]
y = d[:,10]
# print(X.shape)
# print(y.shape)

# We will use an 80/20 split for training, validation and test respectively
# 20% of 19020 =3804
Xt = X[:-3804,:]
yt = y[:-3804]
Xv = X[-3804:,:]
yv = y[-3804:]
# Checking the first 3 lines as well as our output
# print(Xt[:3,:])
# print(yt[:5])
 
cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
model = Sequential()
model.add(Dense(300, input_dim=Xt.shape[1], activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(Xt, yt, epochs=600, batch_size=80, validation_split=0.2, class_weight=cw)
print(model.evaluate(Xv, yv))
# results hover around 0.105 loss and 0.86 accuracy
# I played around with the parameters for 2 hours and the above
# seems the most optimal
# 86 is an 'A' back in Canada so I deemed this accuracy satisfactora