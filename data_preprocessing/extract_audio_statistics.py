import os
import numpy as np
import pandas as pd

## These should be contained within audio_analysis
## loading will consume more memory
#import scipy.stats as stats
#import matplotlib.pyplot as plt
#import soundfile as sf
#import librosa
#import librosa.display as display

import audio_analysis


## Set some global variables
n_fft = 512
use_threshold = None  # threshold power for noise removal 
min_rating = 3.5


### load the metadata
metadata = pd.read_csv('train_metadata.csv')

##for testing purposes 
test_data = metadata.loc[metadata['scientific_name'] == 'Empidonax minimus']


## get all the species conatained in the metadata
## This will be useful if saving an individual file for each species 
bird_labels = pd.unique(metadata.primary_label)
# print(len(bird_labels))

## This is a list containing all the statistical parameters
## index should be identical to the metadata index 
all_stat_params = []


### Initialize file saving options 
save_individual_species = True    # Make false if saving all to a single file 
save_fn = 'all_stat_params.txt'
save_file = open(save_fn,'w')
save_file.write('#filename,mean,variance,skewness,\
kurtosis,mode,gmean,hmean\n')

### iterate over all birds and extract the statistical parameters
for i,row in metadata.iterrows():
    print(i)
    fn = row.filename 
    use_file = os.path.join('train_short_audio',metadata.primary_label[i],fn)

    audio = audio_analysis.Analysis()
    if metadata.rating[i]>=min_rating:
        audio.load_file(use_file,duration=60)
        audio.get_fft()
        audio.get_spectral_params()
        audio.filter_by_threshold(use_threshold,plot=False)
        audio.get_stats_params()

    all_stat_params.append(audio.stat_params)
    #print(row['primary_label'],fn,all_stat_params)
    write_val = ','.join(\
            [str(i) for i in list(audio.stat_params.values() )] ) 

    ### write to a different file for saving memory
    if save_individual_species:
        if row.primary_label != save_fn.split('_')[0]:
            save_file.close()    #First close previous file
            save_fn = row.primary_label+'_stat-params.txt'
            save_file = open(save_fn,'w')
            save_file.write('#filename,mean,variance,skewness,\
kurtosis,mode,gmean,hmean\n')

    save_file.write(','.join([row.primary_label,fn,write_val,'\n']) )
    #audio = None  #To clear memory 

## In case something goes wrong at the end
try:
    save_file.close()
except:
    pass



