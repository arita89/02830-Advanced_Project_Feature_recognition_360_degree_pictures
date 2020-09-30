# -*- coding: utf-8 -*-
"""functions_advanced.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uZZ_Ppk5PfUSXP1xkfzXXiwd-KAV4y1T
"""

from datetime import datetime
import random as rand
from datetime import datetime
import os 
from os import path
import pathlib
from tqdm import tqdm # progress bar
import sys
from sys import exit
from time import time 
import argparse
from sys import exit
import glob 

delimiter = ', ' 
rand.seed(42)
dash = '-'*80

def createFolderInDrive(folder_name, date = None, exercise_number = None ):
  """
  folder_name = string format 'DescriptiveName'
  date = default None, either custom string 'YYYYMMDD' either datetime.today().strftime('%Y%m%d') 
  exercise_number = default None
  """

  root= "/content/drive/My Drive/%s" %(folder_name)
  
  if folder_name.isalpha():

    #os.mkdir ("/content/drive/My Drive/02582-ComputationalDataAnalysis")
    #print (dash)
    if (date is None) & (exercise_number is None):
      
      print ("No date specified, no exercise number specified")
      #print (root)

      while os.path.exists(root) == False:
        print ('%s folder is missing!'%(folder_name))
        os.mkdir (root)
        print ('Folder created in:')
        print (root)
        print (dash)
      if os.path.exists(root) == True:
        print ('%s exist'%(folder_name))
    
    elif (date is not None) & (exercise_number is None): 
      savepath = root+'/'+date

      print ("No exercise number specified")
      #print (savepath)

      while os.path.exists(savepath) == False:
        print ('%s folder is missing!'%(savepath))
        os.mkdir (savepath)
        print ('Folder created in:')
        print (savepath)
        print (dash)
      if os.path.exists(savepath) == True:
        print ('%s exist'%(savepath))

    elif (date is None) & (exercise_number is not None): 
      exercise_root = root+'/'+exercise_number

      print ("No date specified")
      #print (exercise_root)

      while os.path.exists(exercise_root) == False:
        print ('%s folder is missing!'%(exercise_root))
        os.mkdir (exercise_root)
        print ('Folder created in:')
        print (exercise_root)
        print (dash)
      if os.path.exists(exercise_root) == True:
        print ('%s exist'%(exercise_root))

    elif (date is not None) & (exercise_number is not None): 
      
      full_exercise_root = root+'/'+date+'/'+exercise_number
      #print (full_exercise_root)

      while os.path.exists(full_exercise_root) == False:
        print ('%s folder is missing!'%(full_exercise_root))
        os.mkdir (full_exercise_root)
        print ('Folder created in:')
        print (full_exercise_root)
        print (dash)
      if os.path.exists(full_exercise_root) == True:
        print ('%s exist'%(full_exercise_root))
  return

def collect_ipynbs(wdir):
    nbooks = os.popen('find {} -name "*.ipynb" -not -path "*/\.*"'.format(wdir)).read()
    print (type(nbooks))
    if nbooks:
        ##___ clean up ipynb file names
        nbooks = nbooks.replace(" ", "\ ")  # avoid white spaces in a file name
        ipynb_files = nbooks.split('\n')    # split files by new line
        ipynb_files = filter(None, ipynb_files)
        return ipynb_files
    else:
        print("No ipython notebook(s) found in {}!".format(wdir))
        exit(0)

def collect_ipynbs2(wdir):
    nbooks = glob.glob(os.path.join(wdir, "*.ipynb"))
    if nbooks:
        ##___ clean up ipynb file names
        nbooks = nbooks.replace(" ", "\ ")  # avoid white spaces in a file name
        ipynb_files = nbooks.split('\n')    # split files by new line
        ipynb_files = filter(None, ipynb_files)
        return ipynb_files
    else:
        print("Still ...No ipython notebook(s) found in {} ari!".format(wdir))
        exit(0)

def get_format(to):
    available = ['pdf', 'html', 'latex', 'markdown', 'python', 'rst', 'slides']
    if to in available:
        return to
    else:
        print("UNKNOWN TYPE: {}".format(to))
        exit(0)

def change_to_output_dir(wdir, fldr):
    ##___ output directory
    outd = os.path.join(wdir, fldr.upper())
    if not os.path.exists(outd):
        os.makedirs(outd)
    os.chdir(outd)
    return outd

def convert_ipynb(ipynb_files, form, outputdir):
    print("Converting {} notebooks into {} ... \n\n".format( len(ipynb_files), form))
    for nb in ipynb_files:
        if form == 'pdf':
            convert_cmd = 'ipython nbconvert --to latex --post PDF {}'.format(nb)
        else:
            convert_cmd = 'ipython nbconvert --to {} {}'.format(form, nb)
        os.system(convert_cmd)
    print("\nSee output at: {}".format(outputdir))

    


def saveAllFileasFormat(original_dir ,to_dir , format ):
  """
  original_dir = folder where are all the files that needs to be converted to new format
  to_dir = final directory, by default is the same
  format = latex default. can be any  mentionend here https://nbconvert.readthedocs.io/en/latest/usage.html 
  """
  #print ("current directory:")
  #os. getcwd() 
  os.chdir(original_dir)
  print ("reading files from directory: %s" %original_dir)
  #os. getcwd()

  convert_to = get_format(format)
  ipynb_files = collect_ipynbs(original_dir)
  output_dir = change_to_output_dir(original_dir, convert_to)
  convert_ipynb(ipynb_files, convert_to, to_dir)

  #print ("Files in directory to be converted in %s:" %format)
  #file = pathlib.Path("mycfg.py")
  #if file.exists ():
    #!jupyter nbconvert --to latex --config mycfg.py
  #else:
    #print ("mycfg.py not exist")