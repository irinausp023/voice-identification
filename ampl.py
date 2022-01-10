# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 22:49:40 2015

@author: User
"""

# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import glob
import wave
from scipy.signal import butter, filtfilt

from python_speech_features import mfcc
from sklearn.neighbors import KNeighborsClassifier



wavs_train=[]
wavs_test = []

#audio preprocessing
def pre_proc(w):
    frame_rate = w.getframerate()
    fragment_length = w.getnframes()
    block_length = int(fragment_length * frame_rate / 1000)
    samples = np.frombuffer(w.readframes(w.getnframes()), np.dtype(np.int16))
    samples = (4000 * (samples - np.mean(samples)) / np.std(samples)).astype('int16')
    b, a = butter(16, 4000 / (frame_rate / 2))
    samples = filtfilt(b, a, samples).astype('int16')
    
    weights_size = 1000
    weights = np.repeat([1.0], weights_size) / weights_size

    silence_threshold = 600
    smooth = np.convolve(np.abs(samples), weights, 'same')

    samples = samples[smooth > silence_threshold]   
    return samples

#loading training data
for filename in glob.glob("train\*.wav"):
    w = wave.open(filename, 'rb')
    wavs_train.append(pre_proc(w))
#loading testing data
for name in glob.glob("C:test\*.wav"):
    w = wave.open(name, 'rb')
    wavs_test.append(pre_proc(w))  

        
ampl = np.array(wavs_train)
ampl_rasp = np.array(wavs_test)

#function for getting cepstral coefficients
def ceps(X):
    ceps = mfcc(X, winlen=0.025)
    X=[]
    for i in range(0,13):
        X.append(np.mean(ceps[:,i]))
    Vx = np.array(X)
    return Vx

k=len(ampl)

cps = np.zeros((k,13))
kps = np.zeros((k,13))

#getting cepstral coefficients
for i in range(0,k):
    cps[i]=ceps(ampl[i])
    
for i in range(0,k):
    kps[i]=ceps(ampl_rasp[i])
    

labels = np.arange(1, 11, 1)

#identification
model = KNeighborsClassifier(n_neighbors=3,weights='distance',p=2)
model.fit(cps,labels)
print(model)
for i in range(0,k): 
    print(model.predict(kps[[i]]))
