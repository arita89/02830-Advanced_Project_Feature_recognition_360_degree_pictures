# -*- coding: utf-8 -*-
"""__init__.py.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/124m0K-JSlVGd5D3UheDo_-lSThSXXy9h
"""

from datetime import datetime
import random as rand
from datetime import datetime
import os 
from tqdm import tqdm # progress bar
import sys
from time import time 
import argparse

 
delimiter = ', ' 
rand.seed(42)
dash = '-'*80

!apt-get install texlive texlive-xetex texlive-latex-extra pandoc
!pip install pypandoc

from .functions import createFolderInDrive

