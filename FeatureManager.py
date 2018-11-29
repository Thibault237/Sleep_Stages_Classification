# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 09:31:34 2018

@author: Thibault
"""

import pandas as pd
import numpy as np
import h5py
from scipy import signal as sg
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureManager:
    def __init__(self,num_samples = None,path='Sleep_Stages_Classification/Data/train.h5'):
        self.file = h5py.File(path,'r')
        self.data = pd.DataFrame()
        self.labels = pd.Series()
        if num_samples == None:
            num_samples = self.file['eeg_1'].shape[0]
        self.num_samples = num_samples
        self.filter_specs = {'alpha':([8,13],'bandpass'),'beta':([13,22],'bandpass'),
                              'delta':(4,'low'),'theta':([4,8],'bandpass')}
        self.freqs = {'accelerometer_x':10,'accelerometer_y':10,'accelerometer_z':10,
                      'eeg_1':50,'eeg_2':50,'eeg_3':50,'eeg_4':50,'eeg_5':50,'eeg_6':50,'eeg_7':50,
                      'pulse_oximeter_infrared':10}
    
    def butter(self, cutoff, type_butter,f_ech,order=8):
        nyq = 0.5 * f_ech
        if type(cutoff)==list:
            f_cut = []
            for i in range(len(cutoff)):
                f_cut.append(cutoff[i]/nyq)
        else:
            f_cut = cutoff/nyq
        b, a = sg.butter(order, f_cut, btype=type_butter, analog=False)
        return b, a
    
    def get_band(self,band,channel):
        signals = self.file[channel][:self.num_samples,:]
        f_ech = self.freqs.get(channel)
        specs = self.filter_specs.get(band)
        
        b, a = self.butter(*specs,f_ech)
        return sg.lfilter(b, a, signals,1)
    
    def get_esis(self,band,channel,Lambda=100):
        signals = self.get_band(band,channel)
        m = signals.shape[1]
        band_freqs = self.filter_specs[band][0]
        if band == 'delta':
            middle_band = 2
        else:
            middle_band = np.mean(band_freqs)
        
        energie = np.zeros((self.num_samples,))
        for k in range(m):
            energie+=signals[:,k]**2 * middle_band * Lambda
        
        serie = pd.Series(energie,index = range(self.num_samples))
        self.data['esis '+band+' '+channel] = serie
        
    def get_minmax(self,band,channel,Lambda = 100):
        signals = self.get_band(band,channel)
        freq = self.freqs[channel]
        num_fenetres = signals.shape[1]//Lambda
        row_select = np.arange(self.num_samples)
        
        minmax = np.zeros((self.num_samples,))
        for k in range(num_fenetres):
            indices_max = np.argmax(signals[:,k * 100:(k + 1) * 100 - 1],axis=1)+ k * 100
            indices_min = np.argmin(signals[:,k * 100:(k + 1) * 100 - 1],axis=1) + k * 100
            minmax+= np.sqrt((signals[row_select,indices_max] \
                              -signals[row_select,indices_min])**2+((indices_max-indices_min)/freq)**2)
        
        serie = pd.Series(minmax,index = range(self.num_samples))
        self.data['minmax '+band+' '+channel] = serie
    
    def get_argmax_tf(self,channel):
        signals = self.file[channel][:self.num_samples,:]
        tfs = np.abs(np.fft.fft(signals,axis=1))
        num_ech = self.freqs[channel]*30
        maxs = np.argmax(tfs[:,:int(num_ech/2)],axis=1)
        serie = pd.Series(maxs,index = range(self.num_samples))
        self.data['max fft '+channel] = serie
    
    def get_time_Hjorth(self,channel,transform):
        # transform doit être 'activity', 'mobility' ou 'complexity'
        signals = self.file[channel][:self.num_samples,:]
        if transform=='activity':
            res =  np.std(signals,axis=1)**2
        elif transform=='mobility':
            dif = np.diff(signals,n=1,axis=1)
            sigma = np.std(signals,axis=1)
            sigma_dif = np.std(dif,axis=1)
            res = sigma_dif/sigma
        elif transform=='complexity':
            dif = np.diff(signals,n=1,axis=1)
            dif2 = np.diff(signals,n=2,axis=1)
            sigma = np.std(signals,axis=1)
            sigma_dif = np.std(dif,axis=1)
            sigma_dif2 = np.std(dif2,axis=1)
            res = (sigma_dif2/sigma_dif)**2-(sigma_dif/sigma)**2
            
        serie = pd.Series(res,index = range(self.num_samples))
        self.data[transform+' '+channel]=serie
    
    def get_freq_Hjorth(self,channel,transform):
        # transform doit être 'mean' ou 'std'
        signals = self.file[channel][:self.num_samples,:]
        num_ech = self.freqs[channel]*30
        tf = np.abs(np.fft.fft(signals,axis=1))[:,:int(num_ech/2)]
        if transform=='mean':
            res = np.mean(tf,axis=1)
        elif transform=='std':
            res = np.std(tf,axis=1)
        
        serie = pd.Series(res,index = range(self.num_samples))
        self.data['freq '+transform+' '+channel]=serie
    
    def get_all_time_Hjorth(self,transform,excepted=None):
        L=list(self.freqs.keys())
        if excepted!=None:
            a = set(L)
            b = set(excepted)
            L=list(a.difference(b))
        for channel in L:
            self.get_time_Hjorth(channel,transform)
    
    def get_all_freq_Hjorth(self,transform,excepted=None):
        L=list(self.freqs.keys())
        if excepted!=None:
            a = set(L)
            b = set(excepted)
            L=list(a.difference(b))
        for channel in L:
            self.get_freq_Hjorth(channel,transform)
        
    def get_all_esis(self,excepted=None):
        L = range(1,8)
        if excepted!=None:
            a = set(L)
            b = set(excepted)
            L=list(a.difference(b))
        for band in ['alpha','beta','delta','theta']:
            for channel in ['eeg_'+str(i) for i in L]:
                self.get_esis(band,channel)
    
    def get_all_minmax(self,excepted=None):
        L = range(1,8)
        if excepted!=None:
            a = set(L)
            b = set(excepted)
            L=list(a.difference(b))
        for band in ['alpha','beta','delta','theta']:
            for channel in ['eeg_'+str(i) for i in L]:
                self.get_minmax(band,channel)
    
    def get_labels(self,path='Sleep_Stages_Classification/Data/train_y.csv',in_data=False):
        labels = pd.read_csv(path,index_col=0)
        self.labels = pd.Series(labels.iloc[0:self.num_samples,0])
        if in_data==True:
            self.data = pd.concat([self.data,labels],join='inner',axis=1)
    
    def boxplot(self,feature,keep_labels=[0,1,2,3,4]):
        # transform = 'esis' ou 'minmax'
        data = self.data[self.data['sleep_stage'].isin(keep_labels)]
        sns.set(style="whitegrid")
        sns.boxplot(x='sleep_stage',y=feature,data=data\
                        ,showfliers=False)
    
    def histogram(self,transform,band,channel,keep_labels=[0,1,2,3,4],bins=10,FIGSIZE=72,scale=2):
        feature =transform+' '+band+' '+channel
        data = self.data[self.data['sleep_stage'].isin(keep_labels)]
        f,axes = plt.subplots(len(keep_labels),1,figsize=(FIGSIZE,FIGSIZE))
        for i in range(len(keep_labels)):
            axes[i].set_xlim(right=float(scale*np.mean(data[data['sleep_stage']==keep_labels[i]].loc[:,feature])))
        data.hist(column=feature,by='sleep_stage',bins=bins,ax=axes)
    
    def get_features(self):
        return self.data.drop(['sleep_stage'],errors='ignore')
        
    def close(self):
        self.file.close()  
    