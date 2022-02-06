# -*- coding: utf-8 -*-
"""
class for sound files processing

@author: Slawa
"""

import soundfile as sf
import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy import signal, ndimage
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from scipy.stats.mstats import gmean
import pandas as pd
import librosa as ls


class Bird_sound_processor():
    def __init__(self,file=None):
        """file is an audio file path. if None, then a window to chose file will pop up"""
        self.favdata=[]
        self.open(file)
        
    def open(self,file=None):
        """imports an audio file"""
        if file == None:
            file=filedialog.askopenfilename(title='choose sound file')
        self.file=file
        self.FilTdata=[]
        #check that the path is not empty
        if file == '':
            self.data=[]
            self.samplerate=0
            print('wrong file path')
            #!!!ideallly add raise error
        else:
            self.data, self.samplerate = sf.read(file)
            
                
    def FFT_step(self,tdata):
        """computes window fft for Tdata (the lengtht should be a power of 2 for optimum performance)"""
        fdata=fft(tdata,axis=1)
        return fdata
    
    def IFFT_step(self,wdata):
        """computes ifft for wdata"""
        tdata=ifft(wdata,axis=1)
        return tdata
    
    def split(self,Y,size):
        """splits an np.array to a list of arrays with size = size
        if the last array is smaller than size, it is ignored"""
        # comment: an overlap can be added in the splitting process
        N=int(np.floor(len(Y)/size))
        Y1=Y[:N*size]
        return Y1.reshape((N,-1))
    
    def FFT(self,size=2**15,smoothing=True,Gsigma=4):
        #,averaging='arithmetic'
        """splits data in parts with size length each and performs FFT for each of them
        averaging is either geometric or arithmetical
        FYI: size=2**15 is about 1 s at 32000 Hz sampling rate
        smoothing=True to apply gaussian filter to the spectral result
        Gsigma is the standard deviation for Gaussian kernel
        """
        Sdata=self.split(self.data,size)
        Fsdata=self.FFT_step(Sdata)
        AbsFdata=np.abs(Fsdata) #spectral amplitude
        arth_aver=np.mean(AbsFdata,axis=0)
        geom_aver=gmean(AbsFdata,axis=0)
        if np.sum(arth_aver) > np.sum(geom_aver):
            self.favdata=arth_aver
        else:
            self.favdata=geom_aver
            
        if smoothing:
            self.favdata=ndimage.gaussian_filter1d(self.favdata,Gsigma)
        
        Y0=self.favdata.copy()
        Y0/= np.abs(Y0).max()
        Y=Y0[:int(len(Y0)/2)]
        dt=1/self.samplerate #time step
        df=1/dt/(len(Y0)-1) #frequency step
        X=np.arange(len(Y))*df
        self.f=X
        self.If=Y
            
    def filter(self,intervals):
        """bandpass filtering leaving intervals
        intervals=[[f_low1,f_high1],[f_low2,f_high2],...]"""
        
            
    def exportFFT(self,file=None):
        """exports the averaged FFT
        in the form [frequency,data]"""
        X=self.f
        Y=self.If
        np.save(self.file[:-4]+'_averfft.npy',[X,Y])
        
    def loadFFT(self):
        """loads the saved FFT"""
        [X,Y]=np.load(self.file[:-4]+'_averfft.npy')
        self.f=X
        self.If=Y
        
        
    def exportFFTparam(self,Widthlevels=[0.5,np.exp(-2)],Fcut=100):
        """exports FFT parameters"""
        X0=self.f
        Y0=self.If
        ind=X0>Fcut
        X=X0[ind]
        Y=Y0[ind]
        Npeak=Y.argmax()
        Fm=X[Npeak] #peak frequency
        Width=np.zeros(len(Widthlevels))
        StrLevels=''
        for i in range(len(Widthlevels)):
            Nh=np.argwhere(Y[Npeak:]<Widthlevels[i])[0]
            Nl=np.argwhere(Y[:Npeak]<Widthlevels[i])[-1]
            Width[i]=X[Npeak+Nh]-X[Npeak+Nl]
            StrLevels+= ' '+str(Widthlevels[i])
        np.savetxt(self.file[:-4]+'_fftparams.dat', [Fm,Width.reshape((-1,1))],
                        header='peak frequency Hz '+ 
                        'width at levels' + StrLevels
                        ,delimiter='\t',comments='')
    
    def find_main_Tslice(self,Fband,size=2**15,cutlevel=0.06,maxTwindow=6,durationLevel=0.08):
        """find the most pronounced trmporal slice in the Fband=[Fmin,Fmax] frequency window
        finds only one most pronounced feature
        maxTwindow [seconds] is the maximum window to look for
        keep durationLevel > cutlevel"""
        #band pass filtering
        def Ffilter(F,fmin,fmax):
            """filter function"""
            return np.logical_and(F<fmax,F>fmin) #simples option (but should be enough for this purpose)
        Sdata=self.split(self.data,size)
        Fsdata=self.FFT_step(Sdata)
        dt=1/self.samplerate #time step
        df=1/dt/(size-1) #frequency step
        W=np.concatenate((np.arange(int(size/2))*df,np.arange(-int(size/2),0)*df))
        Filter=Ffilter(W,Fband[0],Fband[1])+Ffilter(W,-Fband[1],-Fband[0])
        FilFdata=Fsdata*Filter
        
        plt.figure(4)
        plt.clf()
        plt.yscale('log')
        for i in range(len(Fsdata)):
            plt.plot(np.abs(Fsdata[i]))
        plt.plot(np.abs(FilFdata[0]))
        
        #find a dominant temporal slice
        FilTdata=np.real(self.IFFT_step(FilFdata))
        self.FilTdata=FilTdata
        self.FilPower=np.abs(FilTdata)**2
        Gsigma=int(5*self.samplerate/1000)
        self.FilPowerAv=ndimage.gaussian_filter1d(self.FilPower,Gsigma)
        FS=self.FilPowerAv.reshape((1,-1))[0].copy() #filtered power
        Npeak=FS.argmax() #peak position
        Slice=FS[int(max((0,Npeak-maxTwindow/2/dt))):int(min((len(FS),Npeak+maxTwindow/2/dt)))]
        Slice/=Slice.max()
        ind0=np.argwhere(Slice>cutlevel)
        if len(ind0)>3:
            Slice=Slice[ind0[0][0]:ind0[-1][0]]
        self.bestslice=Slice
        
        plt.figure(5)
        plt.clf()
        t=np.arange(len(Slice))*dt
        plt.plot(t,Slice)
        
        if np.modf(np.log2(len(Slice)))[0] > 0:
            N2=2**int(np.ceil(np.log2(len(Slice)))) #new length =2**
            Slice1=np.concatenate((Slice,np.zeros(N2-len(Slice))))
        else:
            N2=len(Slice)
            Slice1=Slice
            
        #feature duration
        Nspeak=Slice1.argmax()
        if Nspeak==0:
            Nsmin=0
        else:
            if len(np.argwhere(Slice1[:Nspeak]<durationLevel)) == 0:
                Nsmin=0
            else:
                Nsmin=np.argwhere(Slice1[:Nspeak]<durationLevel)[-1][0]
        Nsmax=Nspeak+np.argwhere(Slice1[Nspeak:]<durationLevel)[0][0]
        self.duration_envelope=(Nsmax-Nsmin)*dt
        
        #find typical frequency in the slice
        SF=ndimage.gaussian_filter1d(np.abs(fft(Slice1)),3)
        SF/=SF.max()
        plt.figure(6)
        plt.clf()
        plt.plot(SF)

        
        SFm=np.roll(SF,-1)
        SFp=np.roll(SF,1)
        N12=int(len(SF)/2) # to cut a meaninful half of the spectrum
        ind0=np.logical_and(SF[:N12]>=SFm[:N12],SF[:N12]>=SFp[:N12])
        indm=np.argwhere(ind0==True)[1:,0]
        df2=1/dt/(N2-1) #frequency step
        if len(indm)<2:
            Nmax=0
            self.Fpeak_envelope=0
        else:
            Nmax=SF[indm].argmax()
            self.Fpeak_envelope=df2*(indm[Nmax]-1)
        
            
    def show_T(self,title=None):
        """plots the temporal signal"""
        if len(self.data) == 0:
            print('no data loaded')
        else:
            Y=self.data.copy()
            Y/= np.abs(Y).max()
            X=np.arange(len(Y))/self.samplerate
            ylable='amplitude (normalized)'
            xlable='time (s)'
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(xlable,fontsize=18)
            plt.ylabel(ylable,fontsize=18)
            if not title == None:
                plt.title(title,fontsize=18)
            if len(self.FilTdata)>0:
                Y1=self.FilTdata.reshape((1,-1))[0]
                Y1/= np.abs(Y1).max()
                X1=np.arange(len(Y1))/self.samplerate
                Y2=self.FilPower.reshape((1,-1))[0]
                Y2/= np.abs(Y2).max()
                Y3=self.FilPowerAv.reshape((1,-1))[0]
                Y3/= np.abs(Y3).max()
                plt.plot(X1,Y1,linewidth=3)
                plt.plot(X1,Y2,linewidth=3)
                plt.plot(X1,Y3,linewidth=3)
                plt.legend(('in','filtered','filtered power'))
                
    def show_W(self,title=None,logscale=True):
        """plots the spectral signal"""
        if len(self.favdata) == 0:
            print('no spectral data run .FFT first')
        else:
            Y0=self.favdata.copy()
            Y0/= np.abs(Y0).max()
            Y=Y0[:int(len(Y0)/2)]
            # DT=len(Y)/self.samplerate #time window
            dt=1/self.samplerate #time step
            df=1/dt/(len(Y0)-1) #frequency step
            X=np.arange(len(Y))*df
            ylable='amplitude (normalized)'
            xlable='frequency (Hz)'
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(xlable,fontsize=18)
            plt.ylabel(ylable,fontsize=18)
            if logscale:
                plt.yscale('log')
            if not title == None:
                plt.title(title,fontsize=18)
    
    def convert_export(self,Type='wav'):
        """mostly for my private use
        converts to another data format according to Type using soundfile (see it for supported types)"""
        file2=self.file.split('.')[0]+'.'+Type
        sf.write(file2,self.data,self.samplerate)
        
    def export_params(self,Widthlevels=[0.5,0.8],Fcut=100,save=False):
        """export parameters
        Fcut is the minimum frequency to take into account for FFT"""
        X0=self.f
        Y0=self.If
        ind=X0>Fcut
        X=X0[ind]
        Y=Y0[ind]
        Npeak=Y.argmax()
        Fm=X[Npeak] #peak frequency
        self.Fpeak=Fm
        Width=np.zeros(len(Widthlevels))
        Fmean=np.zeros(len(Widthlevels))
        for i in range(len(Widthlevels)):
            if len(np.argwhere(Y[Npeak:]<Widthlevels[i]))==0:
                Nh=len(Y[Npeak:])-1
            else:
                Nh=np.argwhere(Y[Npeak:]<Widthlevels[i])[0]
            if Npeak == 0:
                Nl=0
            else:
                if len(np.argwhere(Y[:Npeak]<Widthlevels[i]))==0:
                    Nl=0
                else:
                    Nl=np.argwhere(Y[:Npeak]<Widthlevels[i])[-1]
            Width[i]=X[Npeak+Nh]-X[Nl]
            Fmean[i]=(X[Npeak+Nh]+X[Nl])/2
            Centroid=np.sum(Y*X)/np.sum(Y)
            RMS=(np.sum(Y*(X-Centroid)**2)/np.sum(Y))**0.5
            skew=np.sum(Y*(X-Centroid)**3)/np.sum(Y)/RMS**3
            kurtosis=np.sum(Y*(X-Centroid)**4)/np.sum(Y)/RMS**4
            entropy=-np.sum(Y*np.log(Y))/np.log(X[-1]-X[0])
            
        self.width=Width
        self.fmean=Fmean
        
        Titles=['peak frequency Hz','width at level ' + str(Widthlevels[0]),'width at level ' + str(Widthlevels[1]),
                'fmedian at level ' + str(Widthlevels[0]),'fmedian at level ' + str(Widthlevels[1]),
                'pulse duration','pulse rep rate',
                'centroid','RMS width (spread)','skew','kurtosis','entropy']
        values=[Fm,Width[0],Width[1],
                Fmean[0],Fmean[1],
                self.duration_envelope,self.Fpeak_envelope,
                Centroid,RMS,skew,kurtosis,entropy]
        if save:
            np.savetxt(self.file[:-4]+'_params.dat', np.array(values).reshape((1,-1)),
                            header='\t'.join(Titles),
                            delimiter='\t',comments='')
        return [['filename']+Titles,[self.file.split('/')[-1]]+values]


#_____________________________________
    
class sound_subfolder():
    def __init__(self,folder=None,fileslist=[]):
        """folder is an audio folder path. if None, then a window to choose will pop up
        fileslist is the list of full files path for the same bird"""
        self.FFT=[]
        if fileslist==[]:
            self.open(folder)
            self.usefiles=False
            self.birdname=self.folder.split('/')[-1].split('\\')[-1]
        else:
            self.usefiles=True
            self.files=fileslist
            self.birdname=fileslist[0].split('/')[-2]
        print('processing: '+self.birdname)
        
    def open(self,folder=None,file_extention='ogg'):
        """opens and scans the folder for files"""
        if folder == None:
            folder=filedialog.askdirectory(title='folder with sound files')
        self.folder=folder
        files0=list(os.scandir(folder))
        # print(folder)
        
        files=[]
        for f in files0:
            if f.name.split('.')[-1] == file_extention:
                files.append(f.name)
        self.files=files
        # print(files)
        
        
    def FFT_av(self,size=2**15,averaging='arithmetic',postaveraging='geometric'):
        """computes an averaged FFT for all files in the directory and saves it
        size is the data slicing parameter (size of each temporal slice)"""
        FFTs=[]
        for f in self.files:
            if self.usefiles:
                file=f
            else:
                file=self.folder+'/'+f
            BS=Bird_sound_processor(file)
            BS.FFT(size=size)
            BS.exportFFT()
            FFTs.append(BS.If)
        #computing the averaged spectrum of all of the files in the directory
        if averaging == 'geometric':
            FFTfull=gmean(FFTs,axis=0)
        else:
            FFTfull=np.mean(FFTs,axis=0)
        FFTfull/=FFTfull.max()
        if self.usefiles:
            np.save(file.split('.')[0]+'_fullaverfft.npy',[BS.f,FFTfull])
        else:
            # np.save(self.folder+'/'+self.folder.split('/')[-1]+'_fullaverfft.npy',[BS.f,FFTfull])
            np.save(os.path.join(self.folder,self.birdname+'_fullaverfft.npy'),[BS.f,FFTfull])
        self.FFT=FFTfull
        self.f=BS.f
    
    def export_FFTparams(self,Widthlevels=[0.5,0.8],Fcut=100):
        """export FFT parameters
        Fcut is the minimum frequency to take into account"""
        X0=self.f
        Y0=self.FFT
        ind=X0>Fcut
        X=X0[ind]
        Y=Y0[ind]
        Npeak=Y.argmax()
        Fm=X[Npeak] #peak frequency
        self.Fpeak=Fm
        Width=np.zeros(len(Widthlevels))
        Fmean=np.zeros(len(Widthlevels))
        # StrLevels=''
        for i in range(len(Widthlevels)):
            if len(np.argwhere(Y[Npeak:]<Widthlevels[i]))==0:
                Nh=len(Y[Npeak:])-1
            else:
                Nh=np.argwhere(Y[Npeak:]<Widthlevels[i])[0]
            if Npeak == 0:
                Nl=0
            else:
                if len(np.argwhere(Y[:Npeak]<Widthlevels[i]))==0:
                    Nl=0
                else:
                    Nl=np.argwhere(Y[:Npeak]<Widthlevels[i])[-1]
            Width[i]=X[Npeak+Nh]-X[Nl]
            Fmean[i]=(X[Npeak+Nh]+X[Nl])/2
            # StrLevels+= ' '+str(Widthlevels[i])
            
        self.width=Width
        self.fmean=Fmean
        if self.usefiles:
            file='/'.join(self.files[-1].split('/')[:-1])+'/AVfftparams.dat'
        else:
            file=self.folder+'/AVfftparams.dat'
        header='peak frequency Hz '+ '\t width at level' + str(Widthlevels[0])+ '\t width at level' + str(Widthlevels[1])+'\t fmedian at level' + str(Widthlevels[0])+'\t fmedian at level' + str(Widthlevels[1])
        np.savetxt(file, np.concatenate(([Fm],Width.reshape((1,-1))[0],Fmean.reshape((1,-1))[0])).reshape((1,-1)),
                        header=header,
                        delimiter='\t',comments='')
        self.average_output=pd.DataFrame([np.concatenate(([self.birdname],[Fm],Width.reshape((1,-1))[0],Fmean.reshape((1,-1))[0]))],
                                         columns=['primary_label']+header.split('\t'))
        
    def show_W(self,title=None,logscale=True):
        """plots the spectral signal"""
        if len(self.FFT) == 0:
            print('no spectral data run .FFT first')
        else:
            Y=self.FFT
            X=self.f
            ylable='amplitude (normalized)'
            xlable='frequency (Hz)'
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(xlable,fontsize=18)
            plt.ylabel(ylable,fontsize=18)
            if logscale:
                plt.yscale('log')
            if not title == None:
                plt.title(title,fontsize=18)
                
    def rescan_folder(self,outputPath = None,saveoutput = False):
        """go through the folder for th esecond time to save parameters for each """
        Params=[]
        features_df = pd.DataFrame(data=None)
        # target = {}
        for f in self.files:
            if self.usefiles:
                BS=Bird_sound_processor(f)
            else:
                BS=Bird_sound_processor(self.folder+'/'+f)
            BS.loadFFT()
            # print([self.fmean[0]-self.width[0]/2,self.fmean[0]+self.width[0]/2])
            Fband=[self.fmean[0]-self.width[0]/2,self.fmean[0]+self.width[0]/2]
            BS.find_main_Tslice(Fband)
            B=BS.export_params()
            Params.append(B[1])
            #add librosa results
            if self.usefiles:
                BSl=librosa_bird(f)
            else:
                BSl=librosa_bird(self.folder+'/'+f)
            features_dict = BSl.output_spectral_peaks()
            # final_dict = {**features_dict.copy(),**target.copy()}
            final_dict=features_dict
            final_dict['bird name']=self.birdname
            #combine
            for i in range(len(B[0])):
                final_dict[B[0][i]]=B[1][i]
            features_df = features_df.append(final_dict,ignore_index=True)
            print('done with '+f)
            
        # head=B[0]
        # # print(Params)
        # output=pd.DataFrame(Params,columns=head)
        if saveoutput:
            if outputPath == None:
                if self.usefiles:
                    Path=os.path.dirname((os.path.abspath(__file__)))
                    outputPath=Path+'/'+self.birdname+'.csv'
                else:
                    outputPath=self.folder+'/'+self.birdname+'.csv'
            features_df.to_csv(outputPath)
        self.output=features_df
    
            
    def run(self,outputPath=None,saveoutput=False):
        """run functions to find parameters in a folder
        output is the output file path"""
        self.FFT_av()
        self.export_FFTparams()
        self.rescan_folder(outputPath=outputPath,saveoutput=saveoutput)
        return [self.output,self.average_output]

class sound_folder():
    def __init__(self,folder=None,subfolders=[],Number_of_subfolders=0):
        """scan subfolders in a folder
        desired subfolders can be specified in subfolders=[...]
        Number_of_subfolders limits the number of subfolders to be scanned (e.g. for test purposes)
        Number_of_subfolders=0 : scans all subfolders"""
        self.folder=folder
        self.target_subfolders=subfolders
        self.Nsubf=Number_of_subfolders
        if len(subfolders)>0 and folder==None:
            self.folder=os.path.dirname(subfolders[0])
        self.scanforlder()
        # if fileslist==[]:
        #     self.open(folder)
        #     self.usefiles=False
        #     self.birdname=self.folder.split('/')[-1]
        # else:
        #     self.usefiles=True
        #     self.files=fileslist
        #     self.birdname=fileslist[0].split('/')[-2]
            
    def scanforlder(self):
        if self.folder==None:
            self.folder=filedialog.askdirectory(title='folder with subfolders with sound files')
        # self.subfolders=[x[0] for x in os.walk(self.folder)]
        self.subfolders=[]
        for it in os.scandir(self.folder):
            if it.is_dir():
                self.subfolders.append(it)
        # print(self.subfolders)
        
    def run(self,save=True,outputfilename=None):
        """scans the subfolders
        save= True or False defines if save output or not"""
        
        self.output_single = pd.DataFrame(data=None)
        self.output_average = pd.DataFrame(data=None)
        if len(self.target_subfolders)>0:
            folders=[]
            ts=[os.path.realpath(sf).split('\\')[-1] for sf in self.target_subfolders] #target subfolders
            subfolders=[d.name for d in self.subfolders] #all subfolders in the dir
            for subf in ts:
                if subf in subfolders:
                   folders.append(subf) 
        else:
            if self.Nsubf>0:
                folders=self.subfolders[:self.Nsubf]
            else:
                folders=self.subfolders
        for i in range(len(folders)):
            f=folders[i]
            # print(str(f))
            folder=os.path.join(self.folder,f)
            # folder=self.folder+'/'+f
            
            BSF=sound_subfolder(folder)
            [individuals,average]=BSF.run(outputPath=os.path.join(folder,'parameters.csv'),saveoutput=True)
            if i == 0:
                self.output_single = individuals
                self.output_average = average
            else:
                self.output_single = pd.concat([self.output_single,individuals])
                self.output_average = pd.concat([self.output_average,average])
        
        if save:
            if outputfilename == None:
                outPath=filedialog.asksaveasfilename(title='file to save the output',initialdir = self.folder)
            else:
                outPath=os.path.join(self.folder,outputfilename)
            self.output_single.to_csv(outPath+'_single.csv')
            self.output_average.to_csv(outPath+'_average.csv')
        return [self.output_single,self.output_average]
#______________________________________

class librosa_bird():
    
    def __init__(self,file):
        self.file=file
    
    def load_audio_clips(self,audio_fpath):
        #initialize list for holding audio clips
        audio_clips = []
        ext = '.ogg'
        #recursively load audio, assuming all files in directory are audio files
        for root,dirs,files in os.walk(audio_fpath):
            for name in files:
                if name[-4:] == ext:
                    audio_clips.append(root+'/'+name)
            #logging.info('%i audio file(s) in %s have been loaded.'%(len(files),root))
        return audio_clips
    
    def librosa_load_both(self,sr=32000,window='tukey',alpha=0.5,n_fft=2048):
        #librosa takes audio clip path and loads it as a waveform array
        #all audio files in our study are down sampled to 32000 Hz
        x, sr = ls.load(self.file,sr=sr)
        self.sr=sr
        
        #use window function to prevent spectral leakage
        if window == 'tukey':
            window_func = signal.windows.tukey(x.shape[0],alpha)
        else:
            window_func = 1
        x *= window_func
        
        #librosa's short time ft
        X = ls.stft(x,n_fft=n_fft)
        freqs = np.arange(0, 1 + n_fft / 2) * sr / n_fft
        return x, X, sr, freqs
    
        
    def compute_psd_params(self,X_fft):
        Xcov = np.cov(X_fft)
        psd = np.sqrt(np.abs(np.diag(Xcov)))
        self.psd=psd
        psd_mean = np.mean(psd)
        psd_var = np.std(psd)
        return psd, psd_mean, psd_var
    
    def whiten_invert(self,X_fft):
        # estimate covariance matrix
        Xcov = np.cov(X_fft)
        # find power spectral density
        psd = np.sqrt(np.abs(np.diag(Xcov)))
        
        X_noise_norm = X_fft.copy()
        for i in range(len(psd)):
            X_noise_norm[i,:] *= psd[i]/np.mean(psd)
        x_wave = ls.istft(X_noise_norm)
        return x_wave,X_noise_norm,self.sr
    
    def whiten_invert_2(self,X_fft,ntype='thresh_spec',threshold=3):
        # estimate covariance matrix
        Xcov = np.cov(X_fft)
        # find power spectral density
        psd = np.sqrt(np.abs(np.diag(Xcov)))
        X_noise_norm = X_fft.copy()
        
        if ntype=='spec':
            for i in range(len(psd)):
                X_noise_norm[i,:] /= psd[i]
        elif ntype=='mean':
            X_noise_norm /= np.mean(psd)
        elif ntype=='thresh_spec':
            for i in range(len(psd)):
                if psd[i] < np.mean(psd) + threshold * np.std(psd):
                    X_noise_norm[i,:] /= psd[i]
        elif ntype=='thresh_amp':
            X_noise_norm /= np.mean(psd)
            for i in range(len(psd)):
                if psd[i] > np.mean(psd) + threshold * np.std(psd):
                    X_noise_norm[i,:] *= psd[i]*np.mean(psd)
                    
        x_wave = ls.istft(X_noise_norm)
        return x_wave,X_noise_norm,self.sr
    
    def sig_peak_finder_internal(self,input_signal,freqs,primary_widths=[1,200],prominence_limit=0.1,n_peaks=5):
        input_signal_var = np.std(input_signal)
        peaks,peak_features= signal.find_peaks(input_signal/input_signal_var,width=[1,200],prominence=prominence_limit)
        peak_features['indexes'] = peaks
        peak_features['frequency'] = freqs[peaks]
        df = pd.DataFrame(peak_features)
        top_peaks = df.sort_values(df.columns[0],ascending=False,axis=0).head(n_peaks).reset_index()
        feature_dict = {}
        iter_freq = []
        iter_prom = []
        iter_width = []
        iter_acc = []
        iter_index = []
        iter_wh = []
        iterable_features = {}
        
        for index,peak in top_peaks.iterrows():
            ipsd = peak['indexes']
            secondary_width = peak['widths']
            lower_limit = max([0,int(ipsd-secondary_width)])
            upper_limit = min([int(ipsd+secondary_width+1)])
            sig_window = input_signal[lower_limit:upper_limit]/input_signal_var
            peaks = signal.find_peaks_cwt(sig_window,widths=np.arange(1,int(secondary_width+1)),wavelet=signal.ricker)+lower_limit
            acc_check = pd.Series([(np.abs(peaks-ipsd))/secondary_width,0]).explode().values
            accuracy = 1 - min(acc_check)
            
            label_freq = 'peak%i_freqency'%index
            label_prom = 'peak%i_prominence'%index
            label_width = 'peak%i_width'%index
            label_acc = 'peak%i_accuracy'%index
            label_index = 'peak%i_index'%index
            label_wh = 'peak%s_width_height'%index
            
            feature_dict[label_freq] = peak['frequency']
            feature_dict[label_prom] = peak['prominences']
            feature_dict[label_width] = secondary_width
            feature_dict[label_acc] = accuracy
            #feature_dict[label_index] = int(ipsd)
            feature_dict[label_wh] = peak['width_heights']
            
            iter_freq.append(peak['frequency'])
            iter_prom.append(peak['prominences'])
            iter_width.append(secondary_width)
            iter_acc.append(accuracy)
            iter_index.append(int(ipsd))
            iter_wh.append(peak['width_heights'])
        
        iterable_features['frequencies'] = iter_freq
        iterable_features['prominences'] = iter_prom
        iterable_features['widths'] = iter_width
        iterable_features['accuracies'] = iter_acc
        iterable_features['indexes'] = iter_index
        iterable_features['width_heights'] = iter_wh
        
        return feature_dict,iterable_features
    
    def sig_peak_finder(self,input_signal,freqs,primary_widths=[1,200],prominence_limit=0.1,n_peaks=5,label_prefix='peak',extend_label=False,extension='_autocorr'):
        input_signal_var = np.std(input_signal)
        peaks,peak_features= signal.find_peaks(input_signal/input_signal_var,width=[1,200],prominence=prominence_limit)
        peak_features['indexes'] = peaks
        peak_features['frequency'] = freqs[peaks]
        df = pd.DataFrame(peak_features)
        top_peaks = df.sort_values(df.columns[0],ascending=False,axis=0).head(n_peaks).reset_index()
        feature_dict = {}
        
        for index,peak in top_peaks.iterrows():
            ipsd = peak['indexes']
            secondary_width = peak['widths']
            lower_limit = max([0,int(ipsd-secondary_width)])
            upper_limit = min([int(ipsd+secondary_width+1)])
            sig_window = input_signal[lower_limit:upper_limit]/input_signal_var
            peaks = signal.find_peaks_cwt(sig_window,widths=np.arange(1,int(secondary_width+1)),wavelet=signal.ricker)+lower_limit
            acc_check = pd.Series([(np.abs(peaks-ipsd))/secondary_width,0]).explode().values
            accuracy = 1 - min(acc_check)
            
            s_index = label_prefix+str(index)
            if extend_label: s_index+=extension
            
            label_freq = '%s_freqency'%s_index
            label_prom = '%s_prominence'%s_index
            label_width = '%s_width'%s_index
            label_acc = '%s_accuracy'%s_index
            label_index = '%s_index'%s_index
            label_wh = '%s_width_height'%s_index
            
            feature_dict[label_freq] = peak['frequency']
            feature_dict[label_prom] = peak['prominences']
            feature_dict[label_width] = secondary_width
            feature_dict[label_acc] = accuracy
            #feature_dict[label_index] = int(ipsd)
            feature_dict[label_wh] = peak['width_heights']
            
        return feature_dict
    
    def output_spectral_peaks(self,n_peaks_primary=5,n_peaks_secondary=3):
        x,X,sr,freqs = self.librosa_load_both(alpha=0.1)
        psd,psdm,psdv = self.compute_psd_params(X)
        final_dict = {}
        features_dict,iter_feat = self.sig_peak_finder_internal(psd,freqs,n_peaks=n_peaks_primary)
        final_dict.update(features_dict.copy())
        Xcov = np.cov(X)
        for index,peak_index in enumerate(iter_feat['indexes']):
            autocorr_col = np.abs(Xcov[peak_index,:])/psdv
            label_prefix = 'peak%s_autocorr'%str(index)
            autocorr_features = self.sig_peak_finder(autocorr_col,freqs,label_prefix=label_prefix,n_peaks=n_peaks_secondary)
            final_dict.update(autocorr_features)
        return final_dict














#test
# BS=Bird_sound_processor()

# # BS.convert_export()

# plt.figure(2)
# plt.clf()
# BS.FFT()
# BS.show_W()
# # Fband=[3737.4187444685203, 6357.615894039735]
# Fband=[2522.537919248024, 3897.5798821985536]
# BS.find_main_Tslice(Fband)


# plt.figure(1)
# plt.clf()
# BS.show_T()

# BS.export_params()




#folder

# BSF=sound_folder()
# BSF.FFT_av()
# plt.figure(3)
# plt.clf()
# BSF.show_W()
# BSF.export_FFTparams()
# BSF.rescan_folder()

# print([BSF.fmean[0]-BSF.width[0]/2,BSF.fmean[0]+BSF.width[0]/2])