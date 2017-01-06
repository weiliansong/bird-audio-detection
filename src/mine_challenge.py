"""
    This file helps to generate as many confident challenge dataset labels
    as it can.
"""
from __future__ import division, print_function, absolute_import

import util
import numpy as np

nc,dc,rc = util.parse_arguments()
run_name = util.run_name(nc,dc,rc)

checkpoint_dir = 'checkpoint/' + run_name + '/'
submission_file = checkpoint_dir + 'submission.csv'
out_file = checkpoint_dir + 'confident_predictions.csv'

f = open(submission_file, 'r')
f.readline()
f_lines = f.readlines()
f.close()

threshold = 0.90
confident_predictions = []

for line in f_lines:
    tokens = line.strip().split(',')
    if float(tokens[1]) > threshold:
        confident_predictions.append(line)

with open(out_file, 'w') as fb:
    for line in confident_predictions:
        fb.write(line)
