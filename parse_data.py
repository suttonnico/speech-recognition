import scipy.io.wavfile as wavfile
import string
import numpy as np
import argparse

from tensorpack import *
from tensorpack.utils.argtools import memoized
from tensorpack.utils.stats import OnlineMoments
from tensorpack.utils.utils import get_tqdm

import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

from python_speech_features import mfcc

import librosa

CHARSET = set(string.ascii_lowercase + ' ')
PHONEME_LIST = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
    'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

def get_num_in_line(line):
    flag = 0
    for i in range(len(line)):
        if line[i] == ' ':
            if flag == 0:
                space1 = i
                flag = 1
            else:
                space2 = i
    return int(line[0:space1]),int(line[space1+1:space2]),line[space2+1:len(line)-1]

PHONEME_DIC = {v: k for k, v in enumerate(PHONEME_LIST)}
WORD_DIC = {v: k for k, v in enumerate(string.ascii_lowercase + ' ')}

def mfcc_wav(file,w_size,s_size):
    (rate,sig) = wavfile.read(file)
    mfcc_feat = mfcc(sig,rate,winlen=w_size,winstep=s_size,appendEnergy=True)
    return mfcc_feat

def read_timit_txt(f):
    f = open(f, 'r')
    end = 0
    while end < 2:
        text = f.readline()
        print(text)
        print(get_num_in_line(text))
        if text[len(text)-2] == '#':
            end += 1
    #return np.asarray(ret)

(fs, x) = wavfile.read('data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.wav')
plt.figure()
plt.plot(x)


mfcc_feat = mfcc_wav('data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.wav',0.02,0.01)
test = read_timit_txt('data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.PHN')
print(test)
plt.show()