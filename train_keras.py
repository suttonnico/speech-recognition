import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import model_from_json
import pickle
import matplotlib.pyplot as plt

def reshape_data(features,timesteps):
    [N, M] = np.shape(features)
    out = np.zeros([N-timesteps,timesteps,M])
    for i in range(N-timesteps):
        for j in range(timesteps):
            out[i, j, :] = features[i+j, :]
    return out
"""
MOUTH_LIST = ['A'0,'E'1,'O'2,'U'3,'BMP'4,'TS'5,'LN'6,'FV'7,WR8]

PHONEME_LIST = [
    'aa'0, 'ae'1, 'ah'2, 'ao',3 'aw'4, 'ax'5, 'ax-h'6, 'axr'7, 'ay'8, 'b'9, 'bcl'10, 'ch'11, 'd'12, 'dcl'13, 'dh'14,
    'dx'15, 'eh'16, 'el'17, 'em'18, 'en'19, 'eng'20, 'epi'21, 'er'22, 'ey'23, 'f'24, 'g'25, 'gcl'26, 'h#'27, 'hh'28, 'hv'29, 'ih'30,
    'ix'31, 'iy'32, 'jh'33, 'k'34, 'kcl'35, 'l'36, 'm'37, 'n'38, 'ng'39, 'nx'40, 'ow'41, 'oy'42, 'p'43, 'pau'44, 'pcl'45, 'q'46, 'r'47,
    's'48, 'sh'49, 't'50, 'tcl'51, 'th'52, 'uh'53, 'uw'54, 'ux'55, 'v'56, 'w'57, 'y'58, 'z'59, 'zh'60]
"""
def pho2mouth(label):
    nout = np.zeros(9)
    if label[0] == 1:
        nout[0] = 1
    if label[1] == 1:
        nout[0] = 1
    if label[2] == 1:
        nout[0] = 1
    if label[3] == 1:
        nout[0] = 1
    if label[4] == 1:
        nout[0] = 1
    if label[5] == 1:
        nout[0] = 1
    if label[6] == 1:
        nout[0] = 1
    if label[7] == 1:
        nout[0] = 1
    if label[8] == 1:
        nout[0] = 1
    if label[9] == 1:
        nout[4] = 1
    if label[10] == 1:
        nout[4] = 1
    if label[11] == 1:
        nout[5] = 1
    if label[12] == 1:
        nout[6] = 1
    if label[13] == 1:
        nout[6] = 1
    if label[14] == 1:
        nout[6] = 1
    if label[15] == 1:
        nout[6] = 1
    if label[16] == 1:
        nout[1] = 1
    if label[17] == 1:
        nout[1] = 1
    if label[18] == 1:
        nout[1] = 1
    if label[19] == 1:
        nout[1] = 1
    if label[20] == 1:
        nout[1] = 1
    if label[21] == 1:
        nout[4] = 1
    if label[22] == 1:
        nout[1] = 1
    if label[23] == 1:
        nout[1] = 1
    if label[24] == 1:
        nout[7] = 1
    if label[25] == 1:
        nout[1] = 1
    if label[26] == 1:
        nout[1] = 1
    if label[27] == 1:
        nout[4] = 1
    if label[28] == 1:
        nout[1] = 1
    if label[29] == 1:
        nout[1] = 1
    if label[30] == 1:
        nout[1] = 1
    if label[31] == 1:
        nout[1] = 1
    if label[32] == 1:
        nout[1] = 1
    if label[33] == 1:
        nout[1] = 1
    if label[34] == 1:
        nout[1] = 1
    if label[35] == 1:
        nout[1] = 1
    if label[36] == 1:
        nout[6] = 1
    if label[37] == 1:
        nout[4] = 1
    if label[38] == 1:
        nout[6] = 1
    if label[39] == 1:
        nout[6] = 1
    if label[40] == 1:
        nout[6] = 1
    if label[41] == 1:
        nout[2] = 1
    if label[42] == 1:
        nout[2] = 1
    if label[43] == 1:
        nout[4] = 1
    if label[44] == 1:
        nout[5] = 1
    if label[45] == 1:
        nout[5] = 1
    if label[46] == 1:
        nout[3] = 1
    if label[47] == 1:
        nout[8] = 1
    if label[48] == 1:
        nout[5] = 1
    if label[49] == 1:
        nout[5] = 1
    if label[50] == 1:
        nout[5] = 1
    if label[51] == 1:
        nout[5] = 1
    if label[52] == 1:
        nout[5] = 1
    if label[53] == 1:
        nout[3] = 1
    if label[54] == 1:
        nout[3] = 1
    if label[55] == 1:
        nout[3] = 1
    if label[56] == 1:
        nout[7] = 1
    if label[57] == 1:
        nout[8] = 1
    if label[58] == 1:
        nout[1] = 1
    if label[59] == 1:
        nout[5] = 1
    if label[60] == 1:
        nout[5] = 1
    return nout

def pho2mouth_arr(labels):
    out = np.zeros([len(labels),9])
    for i in range(len(labels)):
        out[i] = pho2mouth(labels[i])
    return out



features_train = np.load('features_train.npy')
labels_train = np.load('labels_train.npy')
features_test = np.load('features_test.npy')
labels_test = np.load('labels_test.npy')


print('Changing phonemes to mouths')
labels_train = pho2mouth_arr(labels_train)
labels_test = pho2mouth_arr(labels_test)
print('Done')

data_dim = 39
timesteps = 8
num_classes = 9
print('Reshaping data for LSTM')
features_train = reshape_data(features_train,timesteps)
features_test = reshape_data(features_test,timesteps)
labels_train = labels_train[0:len(labels_train)-timesteps]
labels_test = labels_test[0:len(labels_test)-timesteps]
#labels_test = labels_test[0:1000]
#features_test = features_test[0:1000]
print('Ready')
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(250, return_sequences=True,input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(50, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(250, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(250, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(250))  # return a single vector of dimension 32
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# Generate dummy training data
#x_train = np.random.random((1000, timesteps, data_dim))
#y_train = np.random.random((1000, num_classes))

#print(np.shape(x_train))
#exit(123)
# Generate dummy validation data
#x_val = np.random.random((100, timesteps, data_dim))
#y_val = np.random.random((100, num_classes))

history = model.fit(features_train, labels_train,batch_size=100, epochs=4,validation_data=(features_test, labels_test))

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['acc']
val_acc=history.history['val_acc']
#test
plt.figure()
plt.plot(train_loss,label='loss')
plt.legend(loc='upper right')
plt.title('Train loss 3 layer 250h 8 timesteps')
plt.xlabel('epoch')
plt.ylabel('value')
plt.savefig('TL3l250h8t.png')

plt.figure()
plt.plot(val_acc,label='Accuracy')
plt.legend(loc='upper right')
plt.title('Val_acc 3 layer 250h 8 timesteps')
plt.xlabel('epoch')
plt.ylabel('value')
plt.savefig('ValAcc3l250h8t.png')

plt.figure()
plt.plot(val_loss,label='loss')
plt.legend(loc='upper right')
plt.title('Val_loss 3 layer 250h 8 timesteps')
plt.xlabel('epoch')
plt.ylabel('value')
plt.savefig('Valloss3l250h8t.png')

plt.figure()
plt.plot(train_acc,label='loss')
plt.legend(loc='upper right')
plt.title('Train Accuracy 3 layer 250h 8 timesteps')
plt.xlabel('epoch')
plt.ylabel('value')
plt.savefig('TA3l250h8t.png')


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")










