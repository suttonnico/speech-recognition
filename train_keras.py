import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM


def reshape_data(features,timesteps):
    [N, M] = np.shape(features)
    out = np.zeros([N-timesteps,timesteps,M])
    for i in range(N-timesteps):
        for j in range(timesteps):
            out[i, j, :] = features[i+j, :]
    return out

MOUTH_LIST = ['A','E','O','U']
def pho2mouth(label):
    nout = np.zeros(9)
    if label[0] == 1:
        nout[0] = 1
    if label[1] == 1:
        nout[0] = 1
    if label[0] == 1:
        nout[0] = 1
    if label[0] == 1:
        nout[0] = 1

    return nout

def pho2mouth_arr(labels):
    out = np.zeros([len(labels),9])
    for i in range(len(labels)):
        out[i] = pho2mouth(labels[i])
    return out


PHONEME_LIST = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
    'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


features_train = np.load('features_train.npy')
labels_train = np.load('labels_train.npy')
features_test = np.load('features_test.npy')
labels_test = np.load('labels_test.npy')


data_dim = 13
timesteps = 20
num_classes = 61
print('Reshaping data for LSTM')
features_train = reshape_data(features_train,timesteps)
features_test = reshape_data(features_test,timesteps)
labels_train = labels_train[0:len(labels_train)-timesteps]
labels_test = labels_test[0:len(labels_test)-timesteps]
labels_test = labels_test[0:1000]
features_test = features_test[0:1000]
print('Ready')
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(250, return_sequences=True,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(250, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(250, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(250, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(250))  # return a single vector of dimension 32
model.add(Dense(61, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# Generate dummy training data
#x_train = np.random.random((1000, timesteps, data_dim))
#y_train = np.random.random((1000, num_classes))

#print(np.shape(x_train))
#exit(123)
# Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))

model.fit(features_train, labels_train,batch_size=100, epochs=100,validation_data=(features_test, labels_test))










