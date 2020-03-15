# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:43:44 2020

@author: Alan
"""

import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
import config



side = set(os.listdir(config.side_view))
front = set(os.listdir(config.front_view))


Img_Findings = {}
report_path = config.data_dir

data = []
count = 0
for filename in os.listdir(report_path):
    print('Processing {}'.format(filename))
    
    # create element tree object 
    tree = ET.parse(report_path + filename)

    # get root element 
    root = tree.getroot()

    # Report
    for i, items in enumerate(root.findall('MedlineCitation/Article/Abstract/AbstractText')):
        if i == 2:
            findings = items.text
    
    if findings == None: continue
    
    # Remove invalid tokens, all to lowercase()
    findings = ''.join([c for c in findings.lower() if c.isalpha() or c ==' '])
    findings = findings.replace('xxxx ','')
    
    
    # Image
    side_img = list()
    front_img = list()
    for i, items in enumerate(root.findall('parentImage')):
        id = items.attrib['id']
        id +='.png'        
        if id in side:
            side_img.append(id)
        elif id in front:
            front_img.append(id)
    
    findings = findings.split()
    #print([filename, side_img, front_img, findings])
    data.append([filename, side_img, front_img, findings])
    #if count == 5:
    #    break
    #count+=1
            
df = pd.DataFrame(data, columns=['filename','side','front','findings'])
df.side = df.side.apply(lambda y: np.nan if len(y)==0 else y)
df.front = df.front.apply(lambda y: np.nan if len(y)==0 else y)
df = df.dropna()
df.reset_index(level=0, inplace=True)
df.to_csv('preprocessing/Img_Repor2.csv')
df.to_json('preprocessing/Img_Report2.json',orient='records')
#df = pd.DataFrame.from_dict(Img_Findings, orient='index', columns=['tmp'])
#df['view'], df['Findings'] = df['tmp']
#df = df.dropna()
#df.reset_index(level=0, inplace=True)
#df.to_csv('../data/Img_Report2.csv')
print('--> Data to csv finish')
