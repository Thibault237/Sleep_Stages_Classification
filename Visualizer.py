# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:41:43 2018

@author: Thibault
"""

import numpy as np
import pandas as pd
import h5py
from scipy import signal as sg
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self,path='Data/train.h5'):
        self.file = h5py.File(path,'r')
        self.labels = None
        self.freqs = {'accelerometer_x':10,'accelerometer_y':10,'accelerometer_z':10,
                      'eeg_1':50,'eeg_2':50,'eeg_3':50,'eeg_4':50,'eeg_5':50,'eeg_6':50,'eeg_7':50,
                      'pulse_oximeter_infrared':10}
        self.filter_specs = {'alpha':([8,13],'bandpass'),'beta':([13,22],'bandpass'),
                              'delta':(4,'low'),'theta':([4,8],'bandpass')}
    
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
    
    def get_band(self,band,channel,sample):
        signal = self.file[channel][sample,:]
        f_ech = self.freqs.get(channel)
        specs = self.filter_specs.get(band)
        
        b, a = self.butter(*specs,f_ech)
        return sg.lfilter(b, a, signal)
    
    def get_tf(self,channel,sample):
        signal = self.file[channel][sample,:]
        return np.abs(np.fft.fft(signal, len(signal) * 10))
    
    def get_plot_infos(self,sample,channel,transform=None):
        # sample est le numéro de l'enregistrement, channel vaut 'accelrometer_x','eeg_1',...
        # transform vaut None, 'tf','alpha','beta','delta'...
        if transform == None:
            filtered = self.file[channel][sample,:]
            freq = self.freqs[channel]
            X = np.linspace(0,30,freq*30)
            x_label = 'temps (s)'
            title = channel+' sample: '+str(sample)
            
        elif transform == 'tf':
            filtered = self.get_tf(channel,sample)
            X = np.arange(len(filtered))
            x_label = 'fréquence (Hz)'
            title = transform+' '+channel+' sample: '+str(sample)
        else:
            filtered = self.get_band(transform,channel,sample)
            freq = self.freqs[channel]
            X = np.linspace(0,30,freq*30)
            x_label = 'temps (s)'
            title = transform+' '+channel+' sample: '+str(sample)
        
        return filtered, X, x_label, title
    
    def plot_curve(self,sample,channel,transform=None,figsize=(10,5)):
        # sample est le numéro de l'enregistrement, channel vaut 'accelrometer_x','eeg_1',...
        # transform vaut None, 'tf','alpha','beta','delta'...
        filtered, X, x_label, title = self.get_plot_infos(sample,channel,transform)
        
        f,ax = plt.subplots(figsize=figsize)
        ax.plot(X,filtered),
        ax.set_xlabel(x_label)
        ax.set_title(title)
    
    def compare_transforms(self,sample,channel,transform,figsize=(10,5)):
        
        filtered1, X1, x_label1, title1 = self.get_plot_infos(sample,channel,transform)
        filtered2, X2, x_label2, title2 = self.get_plot_infos(sample,channel,None)
        
        f, (ax1,ax2) = plt.subplots(2,1,figsize=figsize)
        ax1.plot(X1,filtered1),
        ax1.set_xlabel(x_label1)
        ax1.set_title(title1)
        ax2.plot(X2,filtered2),
        ax2.set_xlabel(x_label2)
        ax2.set_title(title2)
        f.tight_layout()
    
    def get_labels(self,path='Data/train_y.csv'):
        self.labels = pd.read_csv(path,index_col=0)
    
    def plot_2curves(self,channel,label1,label2,num_label1,num_label2,transform=None,figsize=(10,10)):
        # fonction pour comparer une transformation appliqué à des observations labellisées différemment
        # channel vaut 'accelrometer_x','eeg_1',...
        # label1 et label2 valent 1,2,3 ou 4
        # num_label1 représente le rang parmis les observations labelisées label1 de la première observation à afficher
        # de même pour num_label2
        
        # self.labels est une dataframe à une colonnes
        # à chaque index i est associé le label de l'observation i
        sample1 = self.labels[self.labels['sleep_stage'] == label1].index[num_label1]
        sample2 = self.labels[self.labels['sleep_stage'] == label2].index[num_label2] 
        
        filtered1, X1, x_label1, title1 = self.get_plot_infos(sample1,channel,transform)
        title1 += ' label: '+str(label1)
        filtered2, X2, x_label2, title2 = self.get_plot_infos(sample2,channel,transform)
        title2 += ' label: '+str(label2)
        
        f, (ax1,ax2) = plt.subplots(2,1,figsize=figsize)
        ax1.plot(X1,filtered1),
        ax1.set_xlabel(x_label1)
        ax1.set_title(title1)
        ax2.plot(X2,filtered2),
        ax2.set_xlabel(x_label2)
        ax2.set_title(title2)
        f.tight_layout()

    def close(self):
        self.file.close()
    