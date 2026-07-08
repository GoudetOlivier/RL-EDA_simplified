import pandas as pd
import numpy as np
import json
#from plotly.subplots import make_subplots

#import plotly.graph_objs as go
from os import walk
import argparse


mypath = "./QUBO/"

new_path = "./new_QUBO/"


all_files = []
for (dirpath, dirnames, filenames) in walk(mypath):
    all_files = filenames
    break;

print(all_files)

for file in all_files:
    
    with open(mypath + file, 'r+') as f:

        data = json.load(f)
        
        data["problem"]['generator'] = "a"
        data["problem"]['author'] = "a"
        
    with open(new_path + file, 'w') as file:
        json.dump(data, file)
            
