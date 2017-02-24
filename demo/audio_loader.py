import tensorflow as tf
import wave
import numpy as np
import random

d = 400000 # number of audio samples for learning

def process_wav(fname):
    try:
        fid = wave.open(fname, "rb")
        raw = fid.readframes(fid.getnframes())
        y = np.fromstring(raw,dtype=np.int16).astype(np.float32)

        # pad if necessary 
        amount_short = 400000-y.size
        if 0 < amount_short:
            y = np.pad(y, 
                    (0,amount_short),
                    'wrap') 

        y = y / 32768.

        y = np.reshape(y, (-1,1,1))
        crop_point = int((len(y)-d-1) * random.random())
        y = y[crop_point:crop_point+d,:,:]
        y = np.reshape(y, (1,400000))

        return y
    except Exception,e:
        print(e)
