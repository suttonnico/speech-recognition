import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import model_from_json
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
from python_speech_features import base


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

def mfcc_wav(file,w_size,s_size):
    (rate,sig) = wavfile.read(file)
    mfcc_feat = mfcc(sig,rate,winlen=w_size,winstep=s_size,appendEnergy=True)
    return mfcc_feat

def take_max(v):
    return np.argmax(v)


def test():
    (fs, x) = wavfile.read('a.wav')
    mfcc_feat = mfcc_wav('a.wav', 0.02, 0.01)
    mfcc_delta = base.delta(mfcc_feat, 2)
    mfcc_delta_delta = base.delta(mfcc_delta, 2)
    mfcc_feat = np.hstack([mfcc_feat, mfcc_delta, mfcc_delta_delta])

    features = mfcc_feat


    data_dim = 39
    timesteps = 8
    num_classes = 9
    print('Reshaping data for LSTM')
    features = reshape_data(features,timesteps)


    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    prediction =  loaded_model.predict(features)
    print(np.shape(prediction))
    print(take_max(prediction))

if __name__ == '__main__':
    print('test')
    test()