#!/usr/bin/env python
from os import path

_dirname = path.dirname(__file__)

data_dir = 'absolute path to the images directory'
vocab_file_path = path.join(_dirname, 'relative path from project root to the pickled vocab file')
img_report_file = path.join(_dirname, 'relative path from project root to the image report csv file')
