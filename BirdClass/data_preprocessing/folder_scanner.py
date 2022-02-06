# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:12:12 2021

@author: Slawa
"""

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)

from Bird_sound_class import sound_folder

"""sound_folder() is taking either a folder argument sound_folder(folder=folder), in that case
the entire folder is scanned,
or a list of folders corresponding  sound_folder(subfolders=[folder1,folder2,...])) """

BSF=sound_folder()
#Number_of_subfolders=... to limit the number of subfolders to scan through
[individuals,average]=BSF.run(outputfilename='features')
#individuals are pandas data frame for single files
#average is the pandas data frame for the averaged spectra across all files
# you can specify the output file name BSF.run(outputfilename='file name')