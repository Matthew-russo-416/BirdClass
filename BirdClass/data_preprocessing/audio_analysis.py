import numpy as np
import os
import scipy.stats as stats

import matplotlib.pyplot as plt

import pandas as pd

import soundfile as sf
import librosa
import librosa.display as display


### After everything works, add error handling 

### Make an object for audio analysis 
class Analysis():

    #set test_run if testing module imports for a new species
    def __init__(self):
        self.stat_params = {}


    ### Only for testing purposes or for some automated tasks 
    def test_run(self,filename,duration=60):
        self.load_file(filename,duration)
        print(self.fileID,'\t',self.duration)
        self.get_fft()
        # self.show_waveplot()
        # self.show_spectrogram()
        
        self.get_onsets()
        self.get_spectral_params()
        self.filter_by_threshold(threshold=None,plot=True,save=True)
            

    ## load the audiofile
    def load_file(self,filename,duration=60):

        ### OS related crap. Add something from os module to clear this 
        if '/' in filename:
            self.fileID = filename.split('.')[0].split('/')[-1]
        elif '\\' in filename:
            self.fileID = filename.split('.')[0].split('\\')[-1]
        else:
            self.fileID = filename.split('.')[0]

        ## actual loading
        self.y,self.sr = librosa.load(filename)
        self.duration = librosa.core.get_duration(self.y, self.sr)
        #print('File loaded. Duration = %.1f s'%self.duration)


    ## get the magnitude and phase parts of fft 
    def get_fft(self,n_fft=512):
        self.D = librosa.stft(self.y) 
        self.S,self.P = librosa.core.magphase(self.D)
        #print('STFT completed')


    ### Display the wave 
    ## add an axis that can be easily implemented into mpl 
    def show_waveplot(self):
        display.waveplot(self.y,self.sr,x_axis='time')
        plt.show()


    ### Display the spectrogram 
    ## add an axis for mpl integration 
    def show_spectrogram(self,axis=None,save=False,savedir=None):
        if axis is None:
            fig,ax = plt.subplots()
        else:
            ax = axis
        spec = display.specshow(librosa.amplitude_to_db(\
                self.D,ref=np.max),y_axis='log',x_axis='time',ax=ax)
        fig.colorbar(spec,format='%2.0f dB',ax=ax)
        plt.tight_layout()
        if save:
            if savedir is None:
                savedir = ''
            fig.savefig(os.path.join([savedir,self.fileID+'_spectrogram.png']))
            plt.close()
        #plt.show()


    ## Determine the audio onsets 
    ## Using the default values here. Change if needed 
    ## This may be useful to cut sounds 
    def get_onsets(self):
        self.onset_strength = librosa.onset.onset_strength(\
                self.y,self.sr,aggregate=np.median,fmax=8000,n_mels=256)
        self.onset_detect = librosa.onset.onset_strength(self.y,self.sr)
        #print('Onsets determined')


    ## Get flattened arrays for frequency, time and intensity
    def get_spectral_params(self):
        try:
            freqs,times,mags = librosa.core.reassigned_spectrogram(\
                    self.y,self.sr,S=self.S)
            self.df = pd.DataFrame() 
            self.df['times'] = np.ravel(times)
            self.df['freqs'] = np.ravel(freqs)
            self.df['mags'] = np.ravel(mags)
            #print('Spectrograph parameters determined')
        except:
            pass


    ## Filter spectrogram by magnitude threshold
    ## This is where the sound is loud 
    ## May not work for all sounds 
    ## Default threshold is half the maximum decibel
    def filter_by_threshold(self,threshold=None,plot=False,save=False,savedir=None):
        if threshold is None:
            threshold = np.log(2)*max(self.df['mags'])
        high_threshold = self.df['mags']>threshold
        self.get_spectral_params()
        high_threshold = self.df['mags']>threshold

        self.threshold_freqs = self.df.freqs[high_threshold]
        self.threshold_mags = self.df.mags[high_threshold]
        self.threshold_times = self.df.times[high_threshold]
        #print('Thresholds determined')

        ## If you want to check the plot 
        if plot:
            fig,ax = plt.subplots(2,1)
            ax[0].set_title(self.fileID)
            ax[0].set_xlabel(r's')
            ax[0].set_ylabel(r'Hz')
            ax[0].set_ylim(0,20000)
            ax[0].scatter(self.threshold_times,self.threshold_freqs,
                    c=self.threshold_mags,s=4)

            ax[1].set_xlabel(r'Hz')
            ax[1].set_ylabel(r'P')
            ax[1].set_xlim(0,20000)
            ax[1].scatter(self.threshold_freqs,self.threshold_mags,
                    c=self.threshold_mags,s=4)
            plt.tight_layout()
            #plt.show()
            if save:
                if savedir is None:
                    savedir = ''
                fig.savefig(os.path.join([savedir,self.fileID+'_thresholdPlots.png']),bb_inches='tight')
                plt.close()


    ## Get statistical parameters after filtering through threshold
    ## For best results, use only peak regions - change appropriate variables within the object and 
    ### rerun the appropriate methods 
    def get_stats_params(self):
        try:
            params = stats.describe(self.threshold_freqs)
            self.stat_params['mean'] = params[2]
            self.stat_params['variance'] = params[3]
            self.stat_params['skewness'] = params[4]
            self.stat_params['kurtosis'] = params[5]
            self.stat_params['mode'] = stats.mode(self.threshold_freqs)[0][0]
            self.stat_params['gmean'] = stats.gmean(self.threshold_freqs)
            self.stat_params['hmean'] = stats.hmean(self.threshold_freqs)
        except:
            pass





#Test here
#use_file = 'train_short_audio/norwat/XC569443.ogg'
#test = Analysis()
#test.test_run(use_file)
#test.load_file(use_file)
#test.get_fft()
## test.show_waveplot()
## test.show_spectrogram()
#
#test.get_onsets()
#test.get_spectral_params()
##test.filter_by_threshold(1,plot=True,save=False)
#print(min(test.df['mags']),max(test.df['mags']))
#
#
#
#
