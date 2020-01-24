# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:43:44 2020

@author: Alan
"""

import numpy as np
import os
import xml.etree.ElementTree as ET 
import pandas as pd


Img_Findings = {}
report_path = 'ecgen-radiology/'
for filename in os.listdir(report_path):
    # create element tree object 
    tree = ET.parse(report_path + filename) 

    # get root element 
    root = tree.getroot() 
    
    # Report
    for i, items in enumerate(root.findall('MedlineCitation/Article/Abstract/AbstractText')):
        if i == 2: 
            findings = items.text
        
    # Image
    for i, items in enumerate(root.findall('parentImage')):
        id = items.attrib['id']
        Img_Findings[id] = findings
        
        
df = pd.DataFrame.from_dict(Img_Findings,orient='index', columns = ['Findings'])
df.reset_index(level=0, inplace=True)
df.to_csv('Img_Report.csv')