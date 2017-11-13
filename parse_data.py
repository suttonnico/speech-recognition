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
import os

from python_speech_features import mfcc

import librosa

CHARSET = set(string.ascii_lowercase + ' ')
PHONEME_LIST = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
    'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

PHONEME_DIC = {v: k for k, v in enumerate(PHONEME_LIST)}
WORD_DIC = {v: k for k, v in enumerate(string.ascii_lowercase + ' ')}


def num2softmax(anot):
    soft_arr = np.zeros([len(anot),61])
    for i in range(len(anot)):
        soft_arr[i,int(anot[i])] = 1
    return soft_arr

def pair_mfcc_with_pho(win_mids,start_ind,end_ind,pho):
    anot_win = np.zeros(len(win_mids))
    ind = 0
    for i in range(len(win_mids)):
        done = 0

        while done == 0:
            #print(win_mids[i],start_ind[ind],end_ind[ind])
            if (win_mids[i] >= start_ind[ind]) & (win_mids[i] <= end_ind[ind]) | (ind == len(end_ind)-1):
                anot_win[i] = pho[ind]
                done = 1
            else:
                ind += 1

    return anot_win


def get_num_in_line(line):
    flag = 0
    for i in range(len(line)):
        if line[i] == ' ':
            if flag == 0:
                space1 = i
                flag = 1
            else:
                space2 = i
    return int(line[0:space1]),int(line[space1+1:space2]),PHONEME_DIC[line[space2+1:len(line)-1]]


def mfcc_wav(file,w_size,s_size):
    (rate,sig) = wavfile.read(file)
    mfcc_feat = mfcc(sig,rate,winlen=w_size,winstep=s_size,appendEnergy=True)
    return mfcc_feat

def read_timit_txt(f):
    f = open(f, 'r')
    end = 0
    start_ind = []
    end_ind = []
    pho = []
    while end < 2:
        text = f.readline()
        nums = get_num_in_line(text)
        start_ind.append(nums[0])
        end_ind.append(nums[1])
        pho.append(nums[2])
        if text[len(text)-2] == '#':
            end += 1
    return start_ind, end_ind, pho


(fs, x) = wavfile.read('data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.wav')



mfcc_feat = mfcc_wav('data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.wav',0.02,0.01)
win_mids = (np.arange(len(mfcc_feat))+1)*fs*0.01
test = read_timit_txt('data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.PHN')

anot = pair_mfcc_with_pho(win_mids,test[0], test[1], test[2])

print('Processing train data... this could take a while')
features = []
labels = []
first = 1
base_train_dir = './data/lisa/data/timit/raw/TIMIT/TRAIN'
DRS = [DR for DR in os.listdir(base_train_dir) ]#if os.path.isfile(f)]
for DR in DRS:
    #print(DR)
    folders = [folder for folder in os.listdir(base_train_dir+'/'+DR)]
    for folder in folders:
        #print(folder)
        files = [f for f in os.listdir(base_train_dir+'/'+DR+'/'+folder)]
        for f in files:
            if f[len(f)-3:len(f)] == 'wav':
                #print(f)
                #print(f[0:len(f)-3]+'PHN')
                pf = f[0:len(f)-3]+'PHN'
                (fs, x) = wavfile.read(base_train_dir+'/'+DR+'/'+folder+'/'+f)
                mfcc_feat = mfcc_wav(base_train_dir+'/'+DR+'/'+folder+'/'+f, 0.02, 0.01)
                win_mids = (np.arange(len(mfcc_feat)) + 1) * fs * 0.01
                [starts,ends,pho] = read_timit_txt(base_train_dir+'/'+DR+'/'+folder+'/'+pf)
                anot = pair_mfcc_with_pho(win_mids, starts, ends, pho)
                soft = num2softmax(anot)
                if first == 1:
                    features = mfcc_feat
                    labels = soft
                    first = 0
                features = np.append(features, mfcc_feat, axis=0)
                labels = np.append(labels, soft, axis=0)

np.save('features_train.npy',features)
np.save('labels_train.npy',labels)

print('Save successful')
print('Processing test data... another while')


features = []
labels = []
first = 1

base_test_dir = './data/lisa/data/timit/raw/TIMIT/TEST'
DRS = [DR for DR in os.listdir(base_train_dir) ]#if os.path.isfile(f)]
for DR in DRS:
    #print(DR)
    folders = [folder for folder in os.listdir(base_test_dir+'/'+DR)]
    for folder in folders:
        #print(folder)
        files = [f for f in os.listdir(base_test_dir+'/'+DR+'/'+folder)]
        for f in files:
            if f[len(f)-3:len(f)] == 'wav':
                #print(f)
                #print(f[0:len(f)-3]+'PHN')
                pf = f[0:len(f)-3]+'PHN'
                (fs, x) = wavfile.read(base_test_dir+'/'+DR+'/'+folder+'/'+f)
                mfcc_feat = mfcc_wav(base_test_dir+'/'+DR+'/'+folder+'/'+f, 0.02, 0.01)
                win_mids = (np.arange(len(mfcc_feat)) + 1) * fs * 0.01
                [starts,ends,pho] = read_timit_txt(base_test_dir+'/'+DR+'/'+folder+'/'+pf)
                anot = pair_mfcc_with_pho(win_mids, starts, ends, pho)
                soft = num2softmax(anot)
                if first == 1:
                    features = mfcc_feat
                    labels = soft
                    first = 0
                features = np.append(features, mfcc_feat, axis=0)
                labels = np.append(labels, soft, axis=0)

np.save('features_test.npy',features)
np.save('labels_test.npy',labels)


print('Save successful')