import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import model_from_json


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

def train():
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


    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    loaded_model.fit(features_train, labels_train,batch_size=100, epochs=1,validation_data=(features_test, labels_test))

    model_json = loaded_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    loaded_model.save_weights("model.h5")
    print("Saved model to disk")

